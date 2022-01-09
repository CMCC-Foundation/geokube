from typing import Iterable


class IndexerDict(dict):
    # TODO: implement union, for instance.
    # current slice is slice(2,7) and in `update` slice is slice(6, 15) result should be slice(2, 15)
    # for the given coordinate/dimension
    def update(self, list_of_dict):
        if not isinstance(list_of_dict, list):
            list_of_dict = [list_of_dict]
        for ld in list_of_dict:
            for k, v in ld.items():
                if isinstance(v, Iterable) and len(v) == 0:
                    continue
                if self.__contains__(k):
                    super().__setitem__(k, [self.__getitem__(k)])
                    if v not in self[k]:
                        self[k].append(v)
                else:
                    super().__setitem__(k, v)
