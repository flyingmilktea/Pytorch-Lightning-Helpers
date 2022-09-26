import warnings
from collections import deque, OrderedDict
from typing import Any, Callable, Deque, Iterator, List, Optional, Sized, Tuple

from pytorch_lightning_helpers.data.datapipes.datapipe import IterDataPipe, functional_datapipe
from pytorch_lightning_helpers.utils import supply_kwargs
from torch.utils.data.datapipes.utils.common import _check_lambda_fn

__all__ = [
    "ConcaterIterDataPipe",
    "DemultiplexerIterDataPipe",
    "ForkerIterDataPipe",
    "MultiplexerIterDataPipe",
    "ZipperIterDataPipe",
]


@functional_datapipe("concat")
class ConcaterIterDataPipe(IterDataPipe):
    r"""
    Concatenates multiple Iterable DataPipes (functional name: ``concat``). The resulting DataPipe will
    yield all the elements from the first input DataPipe, before yielding from the subsequent ones.

    Args:
        datapipes: Iterable DataPipes being concatenated

    Example:
        >>> import random
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1 = IterableWrapper(range(3))
        >>> dp2 = IterableWrapper(range(5))
        >>> list(dp1.concat(dp2))
        [0, 1, 2, 0, 1, 2, 3, 4]
    """
    datapipes: Tuple[IterDataPipe]
    length: Optional[int]

    def __init__(self, *datapipes: IterDataPipe):
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `IterDataPipe`")
        self.datapipes = datapipes  # type: ignore[assignment]
        self.length = None

    def __iter__(self) -> Iterator:
        for dp in self.datapipes:
            for data in dp:
                yield data

    def __len__(self) -> int:
        if self.length is not None:
            if self.length == -1:
                raise TypeError(
                    "{} instance doesn't have valid length".format(type(self).__name__)
                )
            return self.length
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            self.length = sum(len(dp) for dp in self.datapipes)
        else:
            self.length = -1
        return len(self)


@functional_datapipe("fork")
class ForkerIterDataPipe(IterDataPipe):
    r"""
    Creates multiple instances of the same Iterable DataPipe (functional name: ``fork``).

    Args:
        datapipe: Iterable DataPipe being copied
        num_instances: number of instances of the datapipe to create
        buffer_size: this restricts how far ahead the leading child DataPipe
           can read relative to the slowest child DataPipe.
           Defaults to ``1000``. Use ``-1`` for the unlimited buffer.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(5))
        >>> dp1, dp2 = source_dp.fork(num_instances=2)
        >>> list(dp1)
        [0, 1, 2, 3, 4]
        >>> list(dp2)
        [0, 1, 2, 3, 4]
    """

    def __new__(
        cls, datapipe: IterDataPipe, num_instances: int, buffer_size: int = 1000
    ):
        if num_instances < 1:
            raise ValueError(
                f"Expected `num_instaces` larger than 0, but {num_instances} is found"
            )
        if num_instances == 1:
            return datapipe
        container = _ForkerIterDataPipe(datapipe, num_instances, buffer_size)
        return [_ChildDataPipe(container, i) for i in range(num_instances)]


class _ForkerIterDataPipe(IterDataPipe):
    r"""
    Container to hold instance-specific information on behalf of ForkerIterDataPipe. It tracks
    the state of its child DataPipes, maintains the buffer, and yields the next value
    as requested by the child DataPipes.
    """

    def __init__(
        self, datapipe: IterDataPipe, num_instances: int, buffer_size: int = 1000
    ):
        self.main_datapipe = datapipe
        self._datapipe_iterator: Optional[Iterator[Any]] = None
        self.num_instances = num_instances
        self.buffer: Deque = deque()
        self.buffer_size = buffer_size
        if self.buffer_size < 0:
            warnings.warn(
                "Unlimited buffer size is set for `fork`, "
                "please be aware of OOM at random places",
                UserWarning,
            )
        self.child_pointers: List[int] = [
            0
        ] * num_instances  # Indicate the indices of the next element to get
        self.slowest_ptr = 0
        self.leading_ptr = 0
        self.end_ptr: Optional[int] = None

    def __len__(self):
        return len(self.main_datapipe)

    def get_next_element_by_instance(self, instance_id: int):
        if self._datapipe_iterator is None:
            self._datapipe_iterator = iter(self.main_datapipe)
        while self.end_ptr is None or self.child_pointers[instance_id] < self.end_ptr:
            if not self.buffer or self.child_pointers[instance_id] > self.leading_ptr:
                self.leading_ptr = self.child_pointers[instance_id]
                if (
                    self.buffer_size >= 0
                    and self.leading_ptr - self.slowest_ptr + 1 > self.buffer_size
                ):
                    raise BufferError(
                        "ForkerIterDataPipe buffer overflow,"
                        + f"buffer size {self.buffer_size} is insufficient."
                    )
                try:
                    self.buffer.append(next(self._datapipe_iterator))
                    self.child_pointers[instance_id] += 1
                    yield self.buffer[-1]
                except StopIteration:
                    self.end_ptr = self.leading_ptr
            else:  # Child pointer is slower than or equal to the leading_ptr
                buffer_index = self.child_pointers[instance_id] - self.slowest_ptr
                return_val = self.buffer[buffer_index]
                self.child_pointers[instance_id] += 1
                if self.child_pointers[instance_id] - 1 == self.slowest_ptr:
                    new_min = min(
                        self.child_pointers
                    )  # Can optimize by avoiding the call to min()
                    if self.slowest_ptr < new_min:
                        self.slowest_ptr = new_min
                        self.buffer.popleft()
                yield return_val
        if (
            self.end_ptr
            and self.child_pointers[instance_id] == self.end_ptr
            and all(p == self.end_ptr for p in self.child_pointers)
        ):
            self._datapipe_iterator = None

    def is_every_instance_exhausted(self) -> bool:
        return all(self.end_ptr == ptr for ptr in self.child_pointers)

    def reset(self) -> None:
        self._datapipe_iterator = iter(self.main_datapipe)
        self.buffer = deque()
        self.child_pointers = [0] * self.num_instances
        self.slowest_ptr = 0
        self.leading_ptr = 0
        self.end_ptr = None

    def __getstate__(self):
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(self)

        state = (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
        )
        return state

    def __setstate__(self, state):
        (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
        ) = state
        self._datapipe_iterator = None
        self.buffer = deque()
        self.child_pointers = [0] * self.num_instances
        self.slowest_ptr = 0
        self.leading_ptr = 0
        self.end_ptr = None

    def __del__(self):
        self.buffer.clear()

    r"""
    Iterable Datapipe that is a child of a main DataPipe. The instance of this class
    will pass its instance_id to get the next value from its main DataPipe.

    Note:
        ChildDataPipe, like all other IterDataPipe, follows the single iterator per IterDataPipe constraint.
        Since ChildDataPipes share a common buffer, when an iterator is created for one of the ChildDataPipes,
        the previous iterators  for all ChildDataPipes must be invalidated, with the exception when a ChildDataPipe
        hasn't had an iterator created from it since the last invalidation. See the example below.

    Singler Iterator per IteraDataPipe Invalidation Example:
        >>> source_dp = IterableWrapper(range(10))
        >>> cdp1, cdp2 = source_dp.fork(num_instances=2)
        >>> it1, it2 = iter(cdp1), iter(cdp2)
        >>> it3 = iter(cdp1)
        The line above invalidates `it1` and `it2`, and resets `ForkerIterDataPipe`.
        >>> it4 = iter(cdp2)
        The line above doesn't invalidate `it3`, because an iterator for `cdp2` hasn't been created since
        the last invalidation.

    Args:
        main_datapipe: Main DataPipe with a method 'get_next_element_by_instance(instance_id)'
        instance_id: integer identifier of this instance
    """
    _is_child_datapipe: bool = True

    def __init__(self, main_datapipe: IterDataPipe, instance_id: int):
        required_attrs = [
            "get_next_element_by_instance",
            "is_every_instance_exhausted",
            "reset",
        ]
        required_ops = [getattr(main_datapipe, attr) for attr in required_attrs]
        if any(not callable(op) for op in required_ops):
            raise NotImplementedError(
                f"Main Datapipe must have methods {required_attrs} implemented."
            )
        self.main_datapipe: IterDataPipe = main_datapipe
        self.instance_id = instance_id

    def __iter__(self):
        # Note that the logic behind setting iterator ID and `reset` are handled within `hook_iterator`
        # We want to separate the code for reset and yield, so that 'reset' executes before __next__ is called
        return self.main_datapipe.get_next_element_by_instance(self.instance_id)

    def __len__(self):
        return len(self.main_datapipe)

    # This method is called by `hook_iterator` in `_typing.py`.
    def _set_main_datapipe_valid_iterator_id(self) -> int:
        r"""
        Update the valid iterator ID for both this DataPipe object and `main_datapipe`.
        `main_datapipe.reset()` is called when the ID is incremented to a new generation.
        """
        # 1. First time any child iterator is created
        if self.main_datapipe._valid_iterator_id is None:
            self.main_datapipe._valid_iterator_id = 0  # type: ignore[attr-defined]
        # 2. This instance was already in the same generation as `main_datapipe`,
        #    we need to increment the ID further by 1
        elif self.main_datapipe._valid_iterator_id == self._valid_iterator_id:  # type: ignore[has-type]
            self.main_datapipe._valid_iterator_id += 1  # type: ignore[attr-defined]
            # Whenever a new generation of iterator is created, the `main_datapipe` must reset
            if not self.main_datapipe.is_every_instance_exhausted():
                warnings.warn(
                    "Some child DataPipes are not exhausted when __iter__ is called. We are resetting "
                    "the buffer and each child DataPipe will read from the start again.",
                    UserWarning,
                )
            self.main_datapipe.reset()
        # 3. Otherwise, the iterator is behind the others, so it will just need to catch up by setting
        #    the instance's iterator to match that of `main_datapipe`
        self._valid_iterator_id = self.main_datapipe._valid_iterator_id
        return self._valid_iterator_id

    # This method is called by `hook_iterator` in `_typing.py`.
    def _check_valid_iterator_id(self, iterator_id) -> bool:
        r"""
        Check the valid iterator ID against that of DataPipe object and that of `main_datapipe`.
        """
        return (
            iterator_id == self._valid_iterator_id
            and iterator_id == self.main_datapipe._valid_iterator_id
        )


@functional_datapipe("demux")
class DemultiplexerIterDataPipe(IterDataPipe):
    r"""
    Splits the input DataPipe into multiple child DataPipes, using the given
    classification function (functional name: ``demux``). A list of the child DataPipes is returned from this operation.

    Args:
        datapipe: Iterable DataPipe being filtered
        num_instances: number of instances of the DataPipe to create
        classifier_fn: a function that maps values to an integer within the range ``[0, num_instances - 1]`` or ``None``
        drop_none: defaults to ``False``, if ``True``, the function will skip over elements classified as ``None``
        buffer_size: this defines the maximum number of inputs that the buffer can hold across all child
            DataPipes while waiting for their values to be yielded.
            Defaults to ``1000``. Use ``-1`` for the unlimited buffer.

    Examples:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def odd_or_even(n):
        ...     return n % 2
        >>> source_dp = IterableWrapper(range(5))
        >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even)
        >>> list(dp1)
        [0, 2, 4]
        >>> list(dp2)
        [1, 3]
        >>> # It can also filter out any element that gets `None` from the `classifier_fn`
        >>> def odd_or_even_no_zero(n):
        ...     return n % 2 if n != 0 else None
        >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even_no_zero, drop_none=True)
        >>> list(dp1)
        [2, 4]
        >>> list(dp2)
        [1, 3]
    """

    def __new__(
        cls,
        datapipe: IterDataPipe,
        num_instances: int,
        classifier_fn: Callable,
        drop_none: bool = False,
        buffer_size: int = 1000,
    ):
        if num_instances < 1:
            raise ValueError(
                f"Expected `num_instaces` larger than 0, but {num_instances} is found"
            )

        _check_lambda_fn(classifier_fn)

        # When num_instances == 1, demux can be replaced by filter,
        # but keep it as Demultiplexer for the sake of consistency
        # like throwing Error when classification result is out of o range
        container = _DemultiplexerIterDataPipe(
            datapipe, num_instances, classifier_fn, drop_none, buffer_size
        )
        return [_ChildDataPipe(container, i) for i in range(num_instances)]


class _DemultiplexerIterDataPipe(IterDataPipe):
    r"""
    Container to hold instance-specific information on behalf of DemultiplexerIterDataPipe. It tracks
    the state of its child DataPipes, maintains the buffer, classifies and yields the next correct value
    as requested by the child DataPipes.
    """

    def __init__(
        self,
        datapipe: IterDataPipe,
        num_instances: int,
        classifier_fn: Callable,
        drop_none: bool,
        buffer_size: int,
    ):
        self.main_datapipe = datapipe
        self._datapipe_iterator: Optional[Iterator[Any]] = None
        self.num_instances = num_instances
        self.buffer_size = buffer_size
        if self.buffer_size < 0:
            warnings.warn(
                "Unlimited buffer size is set for `demux`, "
                "please be aware of OOM at random places",
                UserWarning,
            )
        self.current_buffer_usage = 0
        self.child_buffers: List = [deque() for _ in range(num_instances)]
        self.classifier_fn = classifier_fn
        self.drop_none = drop_none
        self.main_datapipe_exhausted = False

    def _find_next(self, instance_id: int):
        while True:
            if self.main_datapipe_exhausted:
                raise StopIteration
            if self._datapipe_iterator is None:
                raise ValueError(
                    "_datapipe_iterator has not been set, likely because this private method is called directly "
                    "without invoking get_next_element_by_instance() first."
                )
            value = next(self._datapipe_iterator)
            classification = self.classifier_fn(value)
            if classification is None and self.drop_none:
                continue
            if (
                classification is None
                or classification >= self.num_instances
                or classification < 0
            ):
                raise ValueError(
                    f"Output of the classification fn should be between 0 and {self.num_instances - 1}. "
                    + f"{classification} is returned."
                )
            if classification == instance_id:
                return value
            self.child_buffers[classification].append(value)
            self.current_buffer_usage += 1
            if self.buffer_size >= 0 and self.current_buffer_usage > self.buffer_size:
                raise BufferError(
                    f"DemultiplexerIterDataPipe buffer overflow, buffer size {self.buffer_size} is insufficient."
                )

    def get_next_element_by_instance(self, instance_id: int):
        if self._datapipe_iterator is None and not self.main_datapipe_exhausted:
            self._datapipe_iterator = iter(self.main_datapipe)
        stop = False
        while not stop:
            if self.child_buffers[instance_id]:
                self.current_buffer_usage -= 1
                yield self.child_buffers[instance_id].popleft()
            else:
                try:
                    yield self._find_next(instance_id)
                except StopIteration:
                    stop = True
                    self.main_datapipe_exhausted = True
                    self._datapipe_iterator = None

    def is_every_instance_exhausted(self) -> bool:
        return self.main_datapipe_exhausted and all(
            not child_buffer for child_buffer in self.child_buffers
        )

    def reset(self) -> None:
        self._datapipe_iterator = None
        self.current_buffer_usage = 0
        self.child_buffers = [deque() for _ in range(self.num_instances)]
        self.main_datapipe_exhausted = False

    def __getstate__(self):
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(self)

        state = (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.classifier_fn,
            self.drop_none,
        )
        return state

    def __setstate__(self, state):
        (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.classifier_fn,
            self.drop_none,
        ) = state
        self._datapipe_iterator = None
        self.current_buffer_usage = 0
        self.child_buffers = [deque() for _ in range(self.num_instances)]
        self.main_datapipe_exhausted = False

    def __del__(self):
        for dq in self.child_buffers:
            dq.clear()


class _ChildDataPipe(IterDataPipe):
    r"""
    Iterable Datapipe that is a child of a main DataPipe. The instance of this class
    will pass its instance_id to get the next value from its main DataPipe.

    Note:
        ChildDataPipe, like all other IterDataPipe, follows the single iterator per IterDataPipe constraint.
        Since ChildDataPipes share a common buffer, when an iterator is created for one of the ChildDataPipes,
        the previous iterators  for all ChildDataPipes must be invalidated, with the exception when a ChildDataPipe
        hasn't had an iterator created from it since the last invalidation. See the example below.

    Singler Iterator per IteraDataPipe Invalidation Example:
        >>> source_dp = IterableWrapper(range(10))
        >>> cdp1, cdp2 = source_dp.fork(num_instances=2)
        >>> it1, it2 = iter(cdp1), iter(cdp2)
        >>> it3 = iter(cdp1)
        The line above invalidates `it1` and `it2`, and resets `ForkerIterDataPipe`.
        >>> it4 = iter(cdp2)
        The line above doesn't invalidate `it3`, because an iterator for `cdp2` hasn't been created since
        the last invalidation.

    Args:
        main_datapipe: Main DataPipe with a method 'get_next_element_by_instance(instance_id)'
        instance_id: integer identifier of this instance
    """
    _is_child_datapipe: bool = True

    def __init__(self, main_datapipe: IterDataPipe, instance_id: int):
        required_attrs = [
            "get_next_element_by_instance",
            "is_every_instance_exhausted",
            "reset",
        ]
        required_ops = [getattr(main_datapipe, attr) for attr in required_attrs]
        if any(not callable(op) for op in required_ops):
            raise NotImplementedError(
                f"Main Datapipe must have methods {required_attrs} implemented."
            )
        self.main_datapipe: IterDataPipe = main_datapipe
        self.instance_id = instance_id

    def __iter__(self):
        # Note that the logic behind setting iterator ID and `reset` are handled within `hook_iterator`
        # We want to separate the code for reset and yield, so that 'reset' executes before __next__ is called
        return self.main_datapipe.get_next_element_by_instance(self.instance_id)

    def __len__(self):
        return len(self.main_datapipe)

    # This method is called by `hook_iterator` in `_typing.py`.
    def _set_main_datapipe_valid_iterator_id(self) -> int:
        r"""
        Update the valid iterator ID for both this DataPipe object and `main_datapipe`.
        `main_datapipe.reset()` is called when the ID is incremented to a new generation.
        """
        # 1. First time any child iterator is created
        if self.main_datapipe._valid_iterator_id is None:
            self.main_datapipe._valid_iterator_id = 0  # type: ignore[attr-defined]
        # 2. This instance was already in the same generation as `main_datapipe`,
        #    we need to increment the ID further by 1
        elif self.main_datapipe._valid_iterator_id == self._valid_iterator_id:  # type: ignore[has-type]
            self.main_datapipe._valid_iterator_id += 1  # type: ignore[attr-defined]
            # Whenever a new generation of iterator is created, the `main_datapipe` must reset
            if not self.main_datapipe.is_every_instance_exhausted():
                warnings.warn(
                    "Some child DataPipes are not exhausted when __iter__ is called. We are resetting "
                    "the buffer and each child DataPipe will read from the start again.",
                    UserWarning,
                )
            self.main_datapipe.reset()
        # 3. Otherwise, the iterator is behind the others, so it will just need to catch up by setting
        #    the instance's iterator to match that of `main_datapipe`
        self._valid_iterator_id = self.main_datapipe._valid_iterator_id
        return self._valid_iterator_id

    # This method is called by `hook_iterator` in `_typing.py`.
    def _check_valid_iterator_id(self, iterator_id) -> bool:
        r"""
        Check the valid iterator ID against that of DataPipe object and that of `main_datapipe`.
        """
        return (
            iterator_id == self._valid_iterator_id
            and iterator_id == self.main_datapipe._valid_iterator_id
        )


@functional_datapipe("mux")
class MultiplexerIterDataPipe(IterDataPipe):
    r"""
    Yields one element at a time from each of the input Iterable DataPipes (functional name: ``mux``). As in,
    one element from the 1st input DataPipe, then one element from the 2nd DataPipe in the next iteration,
    and so on. It ends when the shortest input DataPipe is exhausted.

    Args:
        datapipes: Iterable DataPipes that will take turn to yield their elements, until the shortest DataPipe is exhausted

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(3)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
        >>> list(dp1.mux(dp2, dp3))
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
    """

    def __init__(self, *datapipes):
        self.datapipes = datapipes
        self.length: Optional[int] = None
        self.buffer: List = (
            []
        )  # Store values to be yielded only when every iterator provides one

    def __iter__(self):
        iterators = [iter(x) for x in self.datapipes]
        while len(iterators):
            for it in iterators:
                try:
                    value = next(it)
                    self.buffer.append(value)
                except StopIteration:
                    self.buffer.clear()
                    return
            for value in self.buffer:
                yield value
            self.buffer.clear()

    def __len__(self):
        if self.length is not None:
            if self.length == -1:
                raise TypeError(
                    "{} instance doesn't have valid length".format(type(self).__name__)
                )
            return self.length
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            self.length = min(len(dp) for dp in self.datapipes) * len(self.datapipes)
        else:
            self.length = -1
        return len(self)

    def reset(self) -> None:
        self.buffer = []

    def __getstate__(self):
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(self)

        state = (
            self.datapipes,
            self.length,
        )
        return state

    def __setstate__(self, state):
        (
            self.datapipes,
            self.length,
        ) = state
        self.buffer = []

    def __del__(self):
        self.buffer.clear()


@functional_datapipe("zip")
class ZipperIterDataPipe(IterDataPipe):
    r"""
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).
    The output is stopped as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Iterable DataPipes being aggregated

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(5)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
        >>> list(dp1.zip(dp2, dp3))
        [(0, 10, 20), (1, 11, 21), (2, 12, 22), (3, 13, 23), (4, 14, 24)]
    """
    datapipes: Tuple[IterDataPipe]
    length: Optional[int]

    def __init__(self, *datapipes: IterDataPipe):
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError(
                "All inputs are required to be `IterDataPipe` " "for `ZipIterDataPipe`."
            )
        super().__init__()
        self.datapipes = datapipes  # type: ignore[assignment]
        self.length = None

    def __iter__(self) -> Iterator:
        for data in zip(*self.datapipes):
            yield data

    def __len__(self) -> int:
        if self.length is not None:
            if self.length == -1:
                raise TypeError(
                    "{} instance doesn't have valid length".format(type(self).__name__)
                )
            return self.length
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            self.length = min(len(dp) for dp in self.datapipes)
        else:
            self.length = -1
        return len(self)


@functional_datapipe("zip_with_iter")
class IterKeyZipperIterDataPipe(IterDataPipe):
    r"""
    Zips two IterDataPipes together based on the matching key (functional name: ``zip_with_iter``). The keys
    are computed by ``key_fn`` and ``ref_key_fn`` for the two IterDataPipes, respectively. When there isn't a match
    between the elements of the two IterDataPipes, the element from ``ref_datapipe`` is stored in a buffer. Then, the
    next element from ``ref_datapipe`` is tried. After a match is found, the ``merge_fn`` determines how they will
    be combined and returned (a tuple is generated by default).

    Args:
        source_datapipe: IterKeyZipper will yield data based on the order of this IterDataPipe
        ref_datapipe: Reference IterDataPipe from which IterKeyZipper will find items
            with matching key for ``source_datapipe``
        key_fn: Callable function that will compute keys using elements from ``source_datapipe``
        ref_key_fn: Callable function that will compute keys using elements from ``ref_datapipe``
            If it's not specified, the ``key_fn`` will also be applied to elements from ``ref_datapipe``
        keep_key: Option to yield the matching key along with the items in a tuple,
            resulting in `(key, merge_fn(item1, item2))`.
        buffer_size: The size of buffer used to hold key-data pairs from reference DataPipe until a match is found.
            If it's specified as ``None``, the buffer size is set as infinite.
        merge_fn: Function that combines the item from ``source_datapipe`` and the item from ``ref_datapipe``,
            by default a tuple is created

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> from operator import itemgetter
        >>> def merge_fn(t1, t2):
        >>>     return t1[1] + t2[1]
        >>> dp1 = IterableWrapper([('a', 100), ('b', 200), ('c', 300)])
        >>> dp2 = IterableWrapper([('a', 1), ('b', 2), ('c', 3), ('d', 4)])
        >>> res_dp = dp1.zip_with_iter(dp2, key_fn=itemgetter(0),
        >>>                            ref_key_fn=itemgetter(0), keep_key=True, merge_fn=merge_fn)
        >>> list(res_dp)
        [('a', 101), ('b', 202), ('c', 303)]
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        ref_datapipe: IterDataPipe,
        key_fn: Callable,
        ref_key_fn: Optional[Callable] = None,
        keep_key: bool = False,
        buffer_size: int = 10000,
        merge_fn: Optional[Callable] = None,
    ) -> None:
        if not isinstance(ref_datapipe, IterDataPipe):
            raise TypeError(
                f"ref_datapipe must be a IterDataPipe, but its type is {type(ref_datapipe)} instead."
            )
        self.source_datapipe = source_datapipe
        self.ref_datapipe = ref_datapipe
        _check_lambda_fn(key_fn)
        self.key_fn = key_fn
        if ref_key_fn is not None:
            _check_lambda_fn(ref_key_fn)
        self.ref_key_fn = key_fn if ref_key_fn is None else ref_key_fn
        self.keep_key = keep_key
        if merge_fn is not None:
            _check_lambda_fn(merge_fn)
        self.merge_fn = merge_fn
        if buffer_size is not None and buffer_size <= 0:
            raise ValueError(
                "'buffer_size' is required to be either None or a positive integer."
            )
        self.buffer_size: int = buffer_size
        self.buffer: OrderedDict = OrderedDict()

    def __iter__(self) -> Iterator:
        ref_it = iter(self.ref_datapipe)
        warn_once_flag = True
        for data in self.source_datapipe:
            key = supply_kwargs(self.key_fn, data)
            while key not in self.buffer:
                try:
                    ref_data = next(ref_it)
                except StopIteration:
                    raise BufferError(
                        f"No matching key can be found from reference DataPipe for the data {data}. "
                        "Please consider increasing the buffer size."
                    )
                ref_key = supply_kwargs(self.ref_key_fn, ref_data)
                if ref_key in self.buffer:
                    raise ValueError("Duplicate key is found in reference DataPipe")
                if self.buffer_size is not None and len(self.buffer) > self.buffer_size:
                    if warn_once_flag:
                        warn_once_flag = False
                        warnings.warn(
                            "Buffer reaches the upper limit, so reference key-data pair begins to "
                            "be removed from buffer in FIFO order. Please consider increase buffer size."
                        )
                    self.buffer.popitem(last=False)
                self.buffer[ref_key] = ref_data
            res = (
                supply_kwargs(self.merge_fn, {**data, **self.buffer.pop(key)})
                if self.merge_fn
                else (data | self.buffer.pop(key))
            )
            if self.keep_key:
                yield key | res
            else:
                yield res

    def __len__(self) -> int:
        return len(self.source_datapipe)

    def reset(self) -> None:
        self.buffer = OrderedDict()

    def __getstate__(self):
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(self)
        state = (
            self.source_datapipe,
            self.ref_datapipe,
            self.key_fn,
            self.ref_key_fn,
            self.keep_key,
            self.merge_fn,
            self.buffer_size,
        )
        return state

    def __setstate__(self, state):
        (
            self.source_datapipe,
            self.ref_datapipe,
            self.key_fn,
            self.ref_key_fn,
            self.keep_key,
            self.merge_fn,
            self.buffer_size,
        ) = state
        self.buffer = OrderedDict()

    def __del__(self):
        self.buffer.clear()


@functional_datapipe("unbatch")
class UnBatcherIterDataPipe(IterDataPipe):
    r"""
    Undoes batching of data (functional name: ``unbatch``). In other words, it flattens the data up to the specified level
    within a batched DataPipe.

    Args:
        datapipe: Iterable DataPipe being un-batched
        unbatch_level: Defaults to ``1`` (only flattening the top level). If set to ``2``,
            it will flatten the top two levels, and ``-1`` will flatten the entire DataPipe.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper([[[0, 1], [2]], [[3, 4], [5]], [[6]]])
        >>> dp1 = source_dp.unbatch()
        >>> list(dp1)
        [[0, 1], [2], [3, 4], [5], [6]]
        >>> dp2 = source_dp.unbatch(unbatch_level=2)
        >>> list(dp2)
        [0, 1, 2, 3, 4, 5, 6]
    """

    def __init__(self, datapipe: IterDataPipe, unbatch_level: int = 1):
        self.datapipe = datapipe
        self.unbatch_level = unbatch_level

    def __iter__(self):
        for element in self.datapipe:
            for i in self._dive(element, unbatch_level=self.unbatch_level):
                yield i

    def _dive(self, element, unbatch_level):
        if unbatch_level < -1:
            raise ValueError("unbatch_level must be -1 or >= 0")
        if unbatch_level == -1:
            if isinstance(element, list) or isinstance(element, DataChunk):
                for item in element:
                    for i in self._dive(item, unbatch_level=-1):
                        yield i
            else:
                yield element
        elif unbatch_level == 0:
            yield element
        else:
            if isinstance(element, list) or isinstance(element, DataChunk):
                for item in element:
                    for i in self._dive(item, unbatch_level=unbatch_level - 1):
                        yield i
            else:
                raise IndexError(
                    f"unbatch_level {self.unbatch_level} exceeds the depth of the DataPipe"
                )
