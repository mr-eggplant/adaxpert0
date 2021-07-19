import os

import torch

import typing

import torch.distributed as dist


def init(backend="nccl", init_method="env://"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if dist.is_available():
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ['WORLD_SIZE'])

            master_addr = os.environ["MASTER_ADDR"]
            master_port = os.environ["MASTER_PORT"]

            dist.init_process_group(backend=backend,
                                    init_method=init_method,
                                    world_size=world_size,
                                    rank=rank)
            print(f"Init distributed mode(backend={backend}, "
                  f"init_mothod={master_addr}:{master_port}, "
                  f"rank={rank}, pid={os.getpid()}, world_size={world_size}, "
                  f"is_master={is_master()}).")
            return backend, init_method, rank, local_rank, world_size, master_addr, master_port
        else:
            print("Fail to init distributed because torch.distributed is unavailable.")
        return None, None, 0, 0, 1, None, None


def is_dist_avail_and_init():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if is_dist_avail_and_init else 0


def local_rank():
    return int(os.environ["LOCAL_RANK"]) if is_dist_avail_and_init else 0


def world_size():
    return dist.get_world_size() if is_dist_avail_and_init else 1


def is_master():
    return rank() == 0


def save(*args, **kwargs):
    if is_master():
        torch.save(*args, **kwargs)


_str_2_reduceop = dict(
    sum=dist.ReduceOp.SUM,
    mean=dist.ReduceOp.SUM,
    product=dist.ReduceOp.PRODUCT,
    min=dist.ReduceOp.MIN,
    max=dist.ReduceOp.MAX,
    # band=dist.ReduceOp.BAND,
    # bor=dist.ReduceOp.BOR,
    # bxor=dist.ReduceOp.BXOR,
)


def _all_reduce(*args, reduction="sum"):
    t = torch.tensor(args, dtype=torch.float).cuda()
    dist.all_reduce(t, op=_str_2_reduceop[reduction])
    rev = t.tolist()
    if reduction == "mean":
        world_size = dist.get_world_size()
        rev = [item/world_size for item in rev]
    return rev


class Accuracy(object):
    def __init__(self):
        self._is_distributed = dist.is_available() and dist.is_initialized()
        self.reset()

    def reset(self):
        self._n_correct = 0.0
        self._n_total = 0.0
        self._reset_buffer()

    @property
    def rate(self):
        self.sync()
        return self._n_correct / (self._n_total+1e-15)

    @property
    def n_correct(self):
        self.sync()
        return self._n_correct

    @property
    def n_total(self):
        self.sync()
        return self._n_total

    def _reset_buffer(self):
        self._n_correct_since_last_sync = 0.0
        self._n_total_since_last_sync = 0.0
        self._is_synced = True

    def update(self,  n_correct, n_total):
        self._n_correct_since_last_sync += n_correct
        self._n_total_since_last_sync += n_total
        self._is_synced = False

    def sync(self):
        if self._is_synced:
            return
        n_correct = self._n_correct_since_last_sync
        n_total = self._n_total_since_last_sync
        if self._is_distributed:
            n_correct, n_total = _all_reduce(n_correct, n_total, reduction="sum")

        self._n_correct += n_correct
        self._n_total += n_total

        self._reset_buffer()


class AccuracyMetric(object):
    def __init__(self, name: str, topk: typing.Iterable[int] = (1,),):
        self.topk = sorted(list(topk))
        self.reset()

    def reset(self) -> None:
        self.accuracies = [Accuracy() for _ in self.topk]

    def update(self, targets, outputs) -> None:
        maxk = max(self.topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))

        for accuracy, k in zip(self.accuracies, self.topk):
            correct_k = correct[:k].sum().item()
            accuracy.update(correct_k, batch_size)

    def at(self, topk: int) -> Accuracy:
        if topk not in self.topk:
            raise ValueError(f"topk={topk} is not in registered topks={self.topk}")
        accuracy = self.accuracies[self.topk.index(topk)]
        accuracy.sync()
        return accuracy


class AverageMetric(object):
    def __init__(self):
        self._is_distributed = dist.is_available() and dist.is_initialized()
        self.reset()

    def reset(self,) -> None:
        self._n = 0
        self._value = 0.
        self._reset_buf()

    def _reset_buf(self):
        self._n_buf = 0
        self._value_vuf = 0.
        self._is_synced = True

    def sync(self):
        if self._is_synced:
            return
        n = self._n_buf
        value = self._value_vuf
        if self._is_distributed:
            n, value = _all_reduce(n, value)
        self._n += n
        self._value += value
        self._reset_buf()

    def update(self, value) -> None:
        if torch.is_tensor(value):
            self._value_vuf += value.item()
        elif isinstance(value, (int, float)):
            self._value_vuf += value
        else:
            raise ValueError("The parameter 'value' should be int, float or pytorch scalar tensor, but found {}"
                             .format(type(value)))
        self._n_buf += 1
        self._is_synced = False

    def compute(self) -> float:
        self.sync()
        return self._value / (self._n+1e-15)


class AsyncStreamDataLoader():
    def __init__(self, loader, async_stream=True, mean=None, std=None):
        self.loader = iter(loader)
        self.async_stream = async_stream
        if self.async_stream:
            self.stream = torch.cuda.Stream()
        if mean is not None and std is not None:
            self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
            self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)
        else:
            self.mean = None
            self.std = None

        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        
        if self.async_stream:
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.cuda(non_blocking=True)
                self.next_target = self.next_target.cuda(non_blocking=True)

                if self.mean is not None:
                    self.next_input = self.next_input.sub_(self.mean)
                if self.std is not None:
                    self.next_input = self.next_input.div_(self.std)
        else:
            self.next_input = self.next_input.cuda()
            self.next_target = self.next_target.cuda()
            if self.mean is not None:
                self.next_input = self.next_input.sub_(self.mean)
            if self.std is not None:
                self.next_input = self.next_input.div_(self.std)


    def next(self):
        if self.async_stream:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            if input is not None:
                input.record_stream(torch.cuda.current_stream())
            if target is not None:
                target.record_stream(torch.cuda.current_stream())
        else:
            input = self.next_input
            target = self.next_target
        self.preload()
        return input, target
