import torch.nn
from flatten_dict import flatten, unflatten

from utilities import inertia_tensors


class BaseStateObject(torch.nn.Module):

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def set_attribute(self, attr_name, value):
        """
        - means index into list
        : means key into dict
        all means same param for all elements in list
        """
        attr_names = None
        if isinstance(attr_name, list):
            attr_names = [a for a in attr_name[1:]] if len(attr_name) > 2 else attr_name[1]
            attr_name = attr_name[0]

        index, key = None, None
        if "-" in attr_name:
            attr_name, index = attr_name.split('-')
        elif ":" in attr_name:
            attr_name, key = attr_name.split(':')

        if attr_names:
            if index:
                attr = getattr(self, attr_name)
                if index == 'all':
                    for i in range(len(attr)):
                        attr[i].set_attribute(attr_names, value)
                else:
                    attr[int(index)].set_attribute(attr_names, value)
            elif key:
                attr = getattr(self, attr_name)
                if key == 'all':
                    for k in attr.keys():
                        attr[k].set_attribute(attr_names, value)
                else:
                    attr[key].set_attribute(attr_names, value)
            else:
                attr = getattr(self, attr_name)
                attr.set_attribute(attr_names, value)
        else:
            if index:
                attr = getattr(self, attr_name)
                if index == 'all':
                    for i in range(len(attr)):
                        attr[i].set_attribute(attr_names, value)
                else:
                    attr[int(index)].set_attribute(attr_names, value)
            elif key:
                attr = getattr(self, attr_name)
                if key == 'all':
                    for k in attr.keys():
                        attr[k].set_attribute(attr_names, value)
                else:
                    attr[key].set_attribute(attr_names, value)
            else:
                setattr(self, attr_name, value)

    def set_dict_attribute(self, attr_name, value):
        attr_names = attr_name.split("-")
        param_obj, key_lst = attr_names[0], tuple(attr_names[1:])
        dict_attr = getattr(self, param_obj)

        dict_attr = flatten(dict_attr)
        dict_attr[key_lst] = value
        dict_attr = unflatten(dict_attr)

        setattr(self, param_obj, dict_attr)

    @property
    def potential_energy(self):
        return self.mass * self.pos[:, 2:3] * 9.81

    @property
    def kinetic_energy(self):
        lin_vel = self.linear_vel
        lin_ke = 0.5 * self.mass * torch.linalg.vecdot(lin_vel, lin_vel, dim=1).unsqueeze(1)

        ang_vel = self.ang_vel
        I_world = inertia_tensors.body_to_world_torch(self.rot_mat, self.I_body)
        ang_momentum = I_world @ ang_vel
        ang_ke = 0.5 * torch.linalg.vecdot(ang_vel, ang_momentum, dim=1).unsqueeze(1)

        ke = lin_ke + ang_ke

        return ke

    @property
    def total_energy(self):
        return self.potential_energy + self.kinetic_energy