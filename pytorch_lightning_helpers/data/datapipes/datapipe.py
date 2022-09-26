import functools
from typing import Callable, Dict, Optional

from torch.utils.data.datapipes.utils.common import _deprecation_warning
from torch.utils.data.dataset import Dataset, IterableDataset


class functional_datapipe(object):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, cls):
        IterDataPipe.register_datapipe_as_function(self.name, cls)
        return cls


class IterDataPipe(IterableDataset):
    functions = {}
    reduce_ex_hook = None
    getstate_hook = None
    str_hook = None
    repr_hook = None
    _valid_iterator_id = None
    _number_of_samples_yielded = 0
    _fast_forward_iterator = None

    @classmethod
    def register_function(cls, function_name, function):
        cls.functions[function_name] = function

    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register):
        if function_name in cls.functions:
            raise Exception(
                "Unable to add DataPipe function name {} as it is already taken".format(
                    function_name
                )
            )

        def class_function(cls, source_dp, *args, **kwargs):
            result_pipe = cls(source_dp, *args, **kwargs)
            return result_pipe

        function = functools.partial(class_function, cls_to_register)
        cls.functions[function_name] = function

    def __getattr__(self, attribute_name):
        if attribute_name in IterDataPipe.functions:
            function = functools.partial(IterDataPipe.functions[attribute_name], self)
            return function
            # return lambda *args, **kwargs: function(*args, **{k: v for k, v in kwargs.items()
            #                                  if k in getfullargspec(function).args})
        else:
            raise AttributeError(
                "'{0}' object has no attribute '{1}".format(
                    self.__class__.__name__, attribute_name
                )
            )

    def __reduce_ex__(self, *args, **kwargs):
        if IterDataPipe.reduce_ex_hook is not None:
            try:
                return IterDataPipe.reduce_ex_hook(self)
            except NotImplementedError:
                pass
        return super().__reduce_ex__(*args, **kwargs)

    @classmethod
    def set_reduce_ex_hook(cls, hook_fn):
        if IterDataPipe.reduce_ex_hook is not None and hook_fn is not None:
            raise Exception("Attempt to override existing reduce_ex_hook")
        IterDataPipe.reduce_ex_hook = hook_fn

    @classmethod
    def set_getstate_hook(cls, hook_fn):
        if IterDataPipe.getstate_hook is not None and hook_fn is not None:
            raise Exception("Attempt to override existing getstate_hook")
        IterDataPipe.getstate_hook = hook_fn


class MapDataPipe(Dataset):
    r"""
    Map-style DataPipe.

    All datasets that represent a map from keys to data samples should subclass this.
    Subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given, unique key. Subclasses can also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    These DataPipes can be invoked in two ways, using the class constructor or applying their
    functional form onto an existing `MapDataPipe` (recommend, available to most but not all DataPipes).

    Note:
        :class:`~torch.utils.data.DataLoader` by default constructs an index
        sampler that yields integral indices. To make it work with a map-style
        DataPipe with non-integral indices/keys, a custom sampler must be provided.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper, Mapper
        >>> dp = SequenceWrapper(range(10))
        >>> map_dp_1 = dp.map(lambda x: x + 1)  # Using functional form (recommended)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)  # Using class constructor
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> batch_dp = map_dp_1.batch(batch_size=2)
        >>> list(batch_dp)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    """
    functions: Dict[str, Callable] = {}
    reduce_ex_hook: Optional[Callable] = None
    getstate_hook: Optional[Callable] = None
    str_hook: Optional[Callable] = None
    repr_hook: Optional[Callable] = None

    def __getattr__(self, attribute_name):
        if attribute_name in MapDataPipe.functions:
            if attribute_name in _map_deprecated_functional_names:
                kwargs = _map_deprecated_functional_names[attribute_name]
                _deprecation_warning(**kwargs)
            function = functools.partial(MapDataPipe.functions[attribute_name], self)
            return function
        else:
            raise AttributeError(
                "'{0}' object has no attribute '{1}".format(
                    self.__class__.__name__, attribute_name
                )
            )

    @classmethod
    def register_function(cls, function_name, function):
        cls.functions[function_name] = function

    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register):
        if function_name in cls.functions:
            raise Exception(
                "Unable to add DataPipe function name {} as it is already taken".format(
                    function_name
                )
            )

        def class_function(cls, source_dp, *args, **kwargs):
            result_pipe = cls(source_dp, *args, **kwargs)
            return result_pipe

        function = functools.partial(class_function, cls_to_register)
        cls.functions[function_name] = function

    def __getstate__(self):
        """
        This contains special logic to serialize `lambda` functions when `dill` is available.
        If this doesn't cover your custom DataPipe's use case, consider writing custom methods for
        `__getstate__` and `__setstate__`, or use `pickle.dumps` for serialization.
        """
        if MapDataPipe.getstate_hook is not None:
            return MapDataPipe.getstate_hook(self)
        return self.__dict__

    def __reduce_ex__(self, *args, **kwargs):
        if MapDataPipe.reduce_ex_hook is not None:
            try:
                return MapDataPipe.reduce_ex_hook(self)
            except NotImplementedError:
                pass
        return super().__reduce_ex__(*args, **kwargs)

    @classmethod
    def set_getstate_hook(cls, hook_fn):
        if MapDataPipe.getstate_hook is not None and hook_fn is not None:
            raise Exception("Attempt to override existing getstate_hook")
        MapDataPipe.getstate_hook = hook_fn

    @classmethod
    def set_reduce_ex_hook(cls, hook_fn):
        if MapDataPipe.reduce_ex_hook is not None and hook_fn is not None:
            raise Exception("Attempt to override existing reduce_ex_hook")
        MapDataPipe.reduce_ex_hook = hook_fn

    def __repr__(self):
        if self.repr_hook is not None:
            return self.repr_hook(self)
        # Instead of showing <torch. ... .MapperMapDataPipe object at 0x.....>, return the class name
        return str(self.__class__.__qualname__)

    def __str__(self):
        if self.str_hook is not None:
            return self.str_hook(self)
        # Instead of showing <torch. ... .MapperMapDataPipe object at 0x.....>, return the class name
        return str(self.__class__.__qualname__)
