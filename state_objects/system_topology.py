from typing import Dict, List

import torch


class SystemTopology(torch.nn.Module):
    """
    Class to graph of rod and spring attachments.
    Topology dict -> (site name) : [..., Object name, ...]
    Site dict -> (site name) : (world frame coordinate)
    """

    def __init__(self, sites_dict: Dict = None, topology: Dict = None):
        """
        :param sites_dict: dictionary of sites and locations
        :param topology: dictionary of sites to associated object ids
        """
        super().__init__()
        self.sites_dict = sites_dict if sites_dict else {}
        self.topology = topology if topology else {}
        self.sites_dict['None'] = torch.zeros((1, 3, 1), dtype=torch.float64)

    def move_tensors(self, device):
        for k, v in self.topology.items():
            self.topology[k] = v.to(device)

    @classmethod
    def init_to_torch(cls, sites_dict: Dict, topology: Dict, dtype: torch.dtype = torch.float64):
        """
        Method to instantiate system topology with non-tensor inputs
        """
        sites_dict = {k: torch.tensor(v, dtype=dtype).reshape(-1, 3, 1) for k, v in sites_dict.items()}
        return cls(sites_dict, topology)

    def update_site(self, site: str, world_frame_pos: torch.Tensor) -> None:
        """
        Method to update site dict

        :param site: Site name
        :param world_frame_pos: world frame coordinate of site
        """
        self.sites_dict[site] = world_frame_pos
        self.topology[site] = self.topology.get(site, [])

    def update_topology(self, site: str, obj_idx: int):
        """
        Method to update topology dict

        :param site: site name
        :param obj_idx: object idx
        """
        self.topology[site].append(obj_idx)

