from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict

Coord3D = Tuple[int, int, int]

@dataclass(frozen=True)
class Board:
    N: int

    def __init__(self, N):
        self.N = N

    def is_valid_coord(self, i: int, j: int, k: int) -> bool:
        return 1 <= i <= self.N and 1 <= j <= self.N and 1 <= k <= self.N

    def coord_to_id(self, i: int, j: int, k: int) -> int:
        """
        Standard flattening :
        Map triplets (i, j, k) -> id : {0, ..., N^3-1}
        """
        return ((i - 1) * self.N + (j - 1)) * self.N + (k - 1)

    def id_to_coord(self, cell_id: int) -> Coord3D:
        """
        Inverse mapping :
        id -> coordinate
        """
        N = self.N
        i0, rem = divmod(cell_id, N * N) #! quotient and remainder
        j0, k0 = divmod(rem, N)
        return (i0 + 1, j0 + 1, k0 + 1)

    def queen_directions(self, include_vertical: bool = False) -> List[Coord3D]:
        dirs: List[Coord3D] = []

        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    if (a, b, c) == (0, 0, 0):
                        continue
                    if not include_vertical and (a, b, c) in ((0, 0, 1), (0, 0, -1)):
                        continue
                    dirs.append((a, b, c))
        return dirs
    


@dataclass
class LineIndex:
    """
    Precomputes all the attack lines on the 3D board
    """
    geometry: Board
    directions: List[Coord3D]

    #! Filled by build
    lines: List[List[int]] #! Each line is a list of linear cell indices along a straight move line
    cell_to_lines: List[List[int]] #! Inverse mapping : which lines pass through a given cell

    def __init__(self, geometry: Board, include_vertical):
        self.geometry = geometry
        self.directions = geometry.queen_directions(include_vertical=include_vertical)
        self.lines = []
        self.cell_to_lines = [[] for _ in range(geometry.num_cells)]
        self._build()

    def _build(self):
        N = self.geometry.N
        board = self.geometry

        for i in range(1, N + 1):
            for j in range(1, N + 1):
                for k in range(1, N + 1):
                    for (a, b, c) in self.directions:
                        
                        #! A cell is a valid start of line if going backward by the direction vector steps out of the cube 
                        prev_i, prev_j, prev_k = i - a, j - b, k - c
                        if board.is_valid_coord(prev_i, prev_j, prev_k):
                            continue 

                        #! Traces the line
                        cells: List[int] = []
                        cur_i, cur_j, cur_k = i, j, k

                        while board.is_valid_coord(cur_i, cur_j, cur_k):

                            cell_id = board.coord_to_id(cur_i, cur_j, cur_k)
                            cells.append(cell_id)
                            cur_i += a
                            cur_j += b
                            cur_k += c

                        if len(cells) >= 2:

                            line_id = len(self.lines)
                            self.lines.append(cells)
                            for cid in cells:
                                self.cell_to_lines[cid].append(line_id)

    def num_lines(self):
        return len(self.lines)

    def lines_through_cell(self, i: int, j: int, k: int) -> List[int]:
        cid = self.geometry.coord_to_id(i, j, k)
        return self.cell_to_lines[cid]

    def cells_on_line(self, line_id: int) -> List[Coord3D]:
        return [self.geometry.id_to_coord(cid) for cid in self.lines[line_id]]
