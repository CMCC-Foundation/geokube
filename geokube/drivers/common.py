from typing import TYPE_CHECKING, Any, ClassVar

class DriverEntrypoint:
    """
    ``DriverEntrypoint`` is a class container and it is the main interface
    for the driver plugins, see :ref:`RST driver_entrypoint`.
    It shall implement:

    - ``open_field`` method: it shall implement reading from file, variables
      decoding and it returns an instance of :py:class:`~geokube.Field`.
      It shall take in input at least ``filename_or_obj`` argument.
      For more details see :ref:`RST open_field`.
    - ``guess_can_open`` method: it shall return ``True`` if the backend is able to open
      ``filename_or_obj``, ``False`` otherwise. The implementation of this
      method is not mandatory.

    Attributes
    ----------

    open_dataset_parameters : tuple, default: None
        A list of ``open_dataset`` method parameters.
        The setting of this attribute is not mandatory.
    description : str, default: ""
        A short string describing the engine.
        The setting of this attribute is not mandatory.
    url : str, default: ""
        A string with the URL to the backend's documentation.
        The setting of this attribute is not mandatory.
    """

    open_dataset_parameters: ClassVar[tuple | None] = None
    description: ClassVar[str] = ""
    url: ClassVar[str] = ""

    def __repr__(self) -> str:
        txt = f"<{type(self).__name__}>"
        if self.description:
            txt += f"\n  {self.description}"
        if self.url:
            txt += f"\n  Learn more at {self.url}"
        return txt

    def open_cube(
        self,
        filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
        *,
        drop_variables: str | Iterable[str] | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """
        Backend open_dataset method used by Xarray in :py:func:`~xarray.open_dataset`.
        """

        raise NotImplementedError
    
        def open_field(
        self,
        filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
        *,
        drop_variables: str | Iterable[str] | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """
        Backend open_dataset method used by Xarray in :py:func:`~xarray.open_dataset`.
        """

        raise NotImplementedError

    def open_collection(
        self,
        filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
        *,
        drop_variables: str | Iterable[str] | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """
        Backend open_dataset method used by Xarray in :py:func:`~xarray.open_dataset`.
        """

        raise NotImplementedError

    def guess_can_open(
        self,
        filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    ) -> bool:
        """
        Backend open_dataset method used by Xarray in :py:func:`~xarray.open_dataset`.
        """

        return False


# mapping of engine name to (module name, BackendEntrypoint Class)
BACKEND_ENTRYPOINTS: dict[str, tuple[str | None, type[BackendEntrypoint]]] = {}