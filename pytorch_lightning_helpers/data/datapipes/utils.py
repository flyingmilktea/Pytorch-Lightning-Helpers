from pathlib import Path
from typing import Callable, Iterator, List, Sequence, Union

from pytorch_lightning_helpers.data.datapipes.datapipe import IterDataPipe, functional_datapipe
from pytorch_lightning_helpers.utils import supply_kwargs
from torch.utils.data.datapipes.utils.common import get_file_pathnames_from_root
from torchdata.datapipes.iter import IterableWrapper


@functional_datapipe("map")
class MapperIterDataPipe(IterDataPipe):
    datapipe: IterDataPipe
    fn: Callable

    def __init__(
        self,
        datapipe: IterDataPipe,
        fn: Callable,
        input_col=None,
        output_col=None,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe

        self.fn = fn  # type: ignore[assignment]

        self.input_col = input_col
        if input_col is None and output_col is not None:
            raise ValueError("`output_col` must be None when `input_col` is None.")
        if isinstance(output_col, (list, tuple)):
            if len(output_col) > 1:
                raise ValueError("`output_col` must be a single-element list or tuple")
            output_col = output_col[0]
        self.output_col = output_col

    def _apply_fn(self, data):
        return data | supply_kwargs(self.fn, data)

    def __iter__(self) -> Iterator:
        for data in self.datapipe:
            yield self._apply_fn(data)

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError(
            "{} instance doesn't have valid length".format(type(self).__name__)
        )


@functional_datapipe("plh_list_files")
class FileListerIterDataPipe(IterDataPipe):
    r"""
    Given path(s) to the root directory, yields file pathname(s) (path + filename) of files within the root directory.
    Multiple root directories can be provided.

    Args:
        root: Root directory or a sequence of root directories
        masks: Unix style filter string or string list for filtering file name(s)
        recursive: Whether to return pathname from nested directories or not
        abspath: Whether to return relative pathname or absolute pathname
        non_deterministic: Whether to return pathname in sorted order or not.
            If ``False``, the results yielded from each root directory will be sorted
        length: Nominal length of the datapipe

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import FileLister
        >>> dp = FileLister(root=".", recursive=True)
        >>> list(dp)
        ['example.py', './data/data.tar']
    """

    def __init__(
        self,
        root: Union[str, Sequence[str], IterDataPipe] = ".",
        masks: Union[str, List[str]] = "",
        name="path",
        *,
        recursive: bool = False,
        abspath: bool = False,
        non_deterministic: bool = False,
        length: int = -1,
    ) -> None:
        super().__init__()
        if isinstance(root, str):
            root = [
                root,
            ]
        if not isinstance(root, IterDataPipe):
            root = IterableWrapper(root)
        self.datapipe: IterDataPipe = root
        self.masks: Union[str, List[str]] = masks
        self.name = name
        self.recursive: bool = recursive
        self.abspath: bool = abspath
        self.non_deterministic: bool = non_deterministic
        self.length: int = length

    def __iter__(self) -> Iterator[str]:
        for path in self.datapipe:
            yield from (
                {self.name: Path(x)}
                for x in get_file_pathnames_from_root(
                    path,
                    self.masks,
                    self.recursive,
                    self.abspath,
                    self.non_deterministic,
                )
            )

    def __len__(self):
        if self.length == -1:
            raise TypeError(
                "{} instance doesn't have valid length".format(type(self).__name__)
            )
        return self.length


@functional_datapipe("give_name")
class NameGiverIterDataPipe(IterDataPipe):
    def __init__(self, names):
        super().__init__()
        self.names = names

    def __iter__(self):
        for d in self.dp:
            yield {name: v for name, v in zip(self.names, d)}
