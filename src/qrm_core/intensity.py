import numpy as np

class IntensityTable:
    """
        Stores order-flow intensities λ^type(depth, queue_size).
        See Section 2.3.1 of Huang et al. (2015).
    """
    def __init__(
        self,
        max_depth: int,
        max_queue: int,
        types = ("limit", "cancel", "market"),
    ):
        
        self.max_depth = max_depth # K 
        self.max_queue = max_queue # Q
        self.types = types         # order-flow type: (limit, cancel, market)
        self._data = np.zeros((max_depth, max_queue + 1, 2, len(types))) # (depth, queue_size, side, type)
        self._type_index = {t: i for i, t in enumerate(types)} # type -> index
        self._side_index = {side: i for i, side in enumerate(['bid', 'ask'])} # side -> index


    def set_intensity(
        self,
        depth: int,       
        queue: int,       
        side: float,      
        type_name: str,  
        value: float
    ):
        """
            Set intensity for given depth, queue size, side and type.
        """
        d = min(depth, self.max_depth) - 1
        q = min(queue, self.max_queue)
        si = self._side_index[side]
        ti = self._type_index[type_name] 
        self._data[d, q, si, ti] = value

    def get_intensities(
        self,
        depth: int,
        queue: int,
        side: float
    ):
        """
            Return intensities for all types at given depth, queue size and side.
        """
        d = min(depth, self.max_depth) - 1
        q = min(queue, self.max_queue)
        si = self._side_index[side]
        raw = self._data[d, q, si, :]
        return {t: raw[self._type_index[t]] for t in self.types}
    