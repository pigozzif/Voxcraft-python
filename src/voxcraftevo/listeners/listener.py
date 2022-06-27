import abc
from typing import Iterable


class Listener(object):

    def __init__(self, file_path: str, header: Iterable[str], delimiter: str = ";"):
        if "\n" not in header:
            header += "\n"
        self._file = file_path
        self._delimiter = delimiter
        with open(file_path, "w") as file:
            file.write(delimiter.join(header))

    @abc.abstractmethod
    def listen(self, solver) -> None:
        pass
