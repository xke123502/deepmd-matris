import torch
import numpy as np
from copy import deepcopy
import json
# deepmd-kit
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.model.model import BaseModel
from deepmd.pt.utils import env
from deepmd.pt.utils.stat import compute_output_stats
from deepmd.pt.utils.update_sel import UpdateSel
from deepmd.pt.utils.utils import to_numpy_array, to_torch_tensor
from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.utils.path import DPPath
from deepmd.utils.version import check_version_compatibility
from typing import Any, Optional, List, Union, Tuple

from matris.model import MatRIS
from matris.graph import GraphConverter


from matris.graph import GraphConverter, RadiusGraph
from pymatgen.core import Structure, Lattice

def dp_tensor_to_matris_graph(
    coord: torch.Tensor,      # (n_atoms, 3) 
    atype: torch.Tensor,      # (n_atoms,)
    cell: torch.Tensor,       
    type_map: list[str],
    cutoff: float = 6.0,
    three_body_cutoff: float = 4.0,
    graph_id: str = None,
):

    positions = coord.detach().cpu().numpy()
    atom_types = atype.detach().cpu().numpy()
    
    if cell.numel() == 9:
        lattice_vec = cell.view(3, 3).detach().cpu().numpy()
    else:
        lattice_vec = cell.detach().cpu().numpy()
    
    species = [type_map[int(t)] for t in atom_types]
    lattice = Lattice(lattice_vec)
    structure = Structure(
        lattice=lattice,
        species=species,
        coords=positions,
        coords_are_cartesian=True
    )
    
    # 3. Structure → RadiusGraph
    graph_converter = GraphConverter(
        atom_graph_cutoff=cutoff,
        line_graph_cutoff=three_body_cutoff,
    )
    
    graph = graph_converter(structure, graph_id=graph_id)
    return graph


def dp_batch_to_matris_graphs(
    coord: torch.Tensor,      # (batch, n_atoms, 3)
    atype: torch.Tensor,      # (batch, n_atoms)
    cell: torch.Tensor,       # (batch, 9) 
    type_map: list[str],
    cutoff: float = 6.0,
    three_body_cutoff: float = 4.0,
):
    """
    """
    batch_size = coord.shape[0]
    graphs = []
    
    for i in range(batch_size):
        graph = dp_tensor_to_matris_graph(
            coord=coord[i],
            atype=atype[i],
            cell=cell[i] if cell is not None else None,
            type_map=type_map,
            cutoff=cutoff,
            three_body_cutoff=three_body_cutoff,
            graph_id=f"batch_{i}",
        )
        graphs.append(graph)
    
    return graphs



ELEMENTS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

PeriodicTable = {
    **{ee: ii + 1 for ii, ee in enumerate(ELEMENTS)},
    **{f"m{ee}": ii + 1 for ii, ee in enumerate(ELEMENTS)},
    "HW": 1,
    "OW": 8,
}

@BaseModel.register("matris")
class MatRISModel(BaseModel):
    """

    """
    
    def __init__(
        self,
        type_map: list[str],
        sel: int | str,
        # MatRIS parameters
        num_layers: int = 10,
        node_feat_dim: int = 128,
        edge_feat_dim: int = 128,
        three_body_feat_dim: int = 128,
        mlp_hidden_dims: list[int] = [128, 256, 128],
        dropout: float = 0.0,
        use_bias: bool = False,
        distance_expansion: str = "Bessel",
        three_body_expansion: str = "fourier",
        num_radial: int = 7,
        num_angular: int = 7,
        max_l: int = 4,
        max_n: int = 4,
        envelope_exponent: int = 8,
        graph_conv_mlp: str = "gatemlp",
        activation_type: str = "silu",
        norm_type: str = "layer",
        pairwise_cutoff: float = 6.0,
        three_body_cutoff: float = 4.5,
        use_smoothed_for_delta_edge: bool = True,
        learnable_basis: bool = True,
        is_intensive: bool = True,  # True for per-atom energy, used with e0 bias
        is_conservation: bool = True,
        reference_energy: str | None = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        
        # Store all parameters
        self.params = {
            "type_map": type_map,
            "sel": sel,
            "num_layers": num_layers,
            "node_feat_dim": node_feat_dim,
            "edge_feat_dim": edge_feat_dim,
            "three_body_feat_dim": three_body_feat_dim,
            "mlp_hidden_dims": mlp_hidden_dims,
            "dropout": dropout,
            "use_bias": use_bias,
            "distance_expansion": distance_expansion,
            "three_body_expansion": three_body_expansion,
            "num_radial": num_radial,
            "num_angular": num_angular,
            "max_l": max_l,
            "max_n": max_n,
            "envelope_exponent": envelope_exponent,
            "graph_conv_mlp": graph_conv_mlp,
            "activation_type": activation_type,
            "norm_type": norm_type,
            "pairwise_cutoff": pairwise_cutoff,
            "three_body_cutoff": three_body_cutoff,
            "use_smoothed_for_delta_edge": use_smoothed_for_delta_edge,
            "learnable_basis": learnable_basis,
            "is_intensive": is_intensive,
            "is_conservation": is_conservation,
            "reference_energy": reference_energy,
        }
        
        self.type_map = type_map
        self.ntypes = len(type_map)
        self.rcut = pairwise_cutoff  # MatRIS uses pairwise_cutoff
        self.three_body_cutoff = three_body_cutoff
        self.sel = sel
        self.preset_out_bias: dict[str, list] = {"energy": []}
        self.mm_types = []
        
        # Handle MM types
        atomic_numbers = []
        for ii, tt in enumerate(type_map):
            atomic_numbers.append(PeriodicTable[tt])
            if not tt.startswith("m") and tt not in {"HW", "OW"}:
                self.preset_out_bias["energy"].append(None)
            else:
                self.preset_out_bias["energy"].append([0])
                self.mm_types.append(ii)
        
        self.atomic_numbers = atomic_numbers
        
        # filter MatRIS param（type_map, sel）
        matris_params = {k: v for k, v in self.params.items() 
                        if k not in ['type_map', 'sel']}

        self.graph_converter = GraphConverter(
            atom_graph_cutoff=self.rcut,
            line_graph_cutoff=self.three_body_cutoff,
        ) # 
        self.matris_model = MatRIS(**matris_params)
        self.matris_model = self.matris_model.to(torch.float32)
        # Register energy bias buffer
        self.register_buffer(
            "e0",
            torch.zeros(
                self.ntypes,
                dtype=env.GLOBAL_PT_ENER_FLOAT_PRECISION,
                device=env.DEVICE,
            ),
        )
    
    @torch.jit.export
    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        ) -> dict[str, torch.Tensor]:
        """Forward pass using gradient-preserving batch processing."""
        if fparam is not None:
            raise ValueError("fparam is unsupported")
        if aparam is not None:
            raise ValueError("aparam is unsupported")
                    
        nf, nloc = atype.shape
        coord = coord.view(nf, nloc, 3)
        coord = coord.to(torch.float32)
        #import time
        # t0 = time.time()
        batch_graph = dp_batch_to_matris_graphs(
            coord, atype, box, self.type_map, self.rcut, self.three_body_cutoff
        )
        # t1 = time.time()
        # print(f"graph convert time cost: {t1 - t0} seconds")
        device = coord.device
        batch_graph = [g.to(device) for g in batch_graph] # 
        
        # run sevennet model
        #t2 = time.time()
        result = self.matris_model(batch_graph, task="ef", is_training=self.training)
        #t3 = time.time()
        #print(f"matris model forward time cost: {t3 - t2} seconds")
        if self.params["is_intensive"]: # per-atom energy
            energy_per_atom = result["e"]
            atoms_per_graph = result['atoms_per_graph']
            atom_energy_batch = energy_per_atom.unsqueeze(1).repeat(1, nloc)
        else:
            total_energy = result['e']  # [nf]
            atoms_per_graph = result['atoms_per_graph']
            energy_per_atom = total_energy / atoms_per_graph
            atom_energy_batch = energy_per_atom.unsqueeze(1).repeat(1, nloc)
        
        # add energy bias (e0): reference energy per atom type
        bias_expanded = self.e0[atype].to(atom_energy_batch.dtype).to(atom_energy_batch.device)
        atom_energy_batch = atom_energy_batch + bias_expanded
        
        # compute total energy
        energy_batch = torch.sum(atom_energy_batch, dim=1).view(nf, 1)
        
        
        # Process force: MatRIS returns List[Tensor([n_i, 3])], need to convert to [nf, nloc, 3]
        forces_list = result['f']  # List[Tensor]
        # Concatenate all forces and reshape
        force_concat = torch.cat(forces_list, dim=0)  # [total_atoms, 3]
        # Reshape to batch format
        force_batch = force_concat.view(nf, nloc, 3)  # [nf, nloc, 3]

        
        # compute virial tensor
        coord_for_virial = coord.view(nf, nloc, 3)
        virial_batch = torch.sum(
            force_batch.unsqueeze(-1) @ coord_for_virial.unsqueeze(-2), 
            dim=1
        ).view(nf, 9)
        
        # prepare model output
        model_predict = {
            "atom_energy": atom_energy_batch.unsqueeze(-1),  # (nf, nloc, 1)
            "energy": energy_batch,                         # (nf, 1)
            "force": force_batch,                           # (nf, nloc, 3)
            "virial": virial_batch,                         # (nf, 9)
        }
        #do_atomic_virial = True
        if do_atomic_virial:
            # compute atomic virial
            atomic_virial = force_batch.unsqueeze(-1) @ coord_for_virial.unsqueeze(-2)
            model_predict["atom_virial"] = atomic_virial.view(nf, nloc, 9)
        
        return model_predict
    
    # implement all necessary BaseModel methods (same as mace.py)
    @torch.jit.export
    def fitting_output_def(self) -> FittingOutputDef:
        return FittingOutputDef([
            OutputVariableDef(
                name="energy",
                shape=[1],
                reducible=True,
                r_differentiable=True,
                c_differentiable=True,
            ),
        ])
    
    @torch.jit.export
    def get_rcut(self) -> float:
        return self.rcut
    
    @torch.jit.export
    def get_type_map(self) -> list[str]:
        return self.type_map
    
    @torch.jit.export
    def get_sel(self) -> list[int]:
        if isinstance(self.sel, int):
            return [self.sel]
        else:
            return [120]
    
    @torch.jit.export
    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return 0

    @torch.jit.export
    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return 0

    @torch.jit.export
    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return []

    @torch.jit.export
    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False

    @torch.jit.export
    def mixed_types(self) -> bool:
        """Return whether the model is in mixed-types mode.

        If true, the model
        1. assumes total number of atoms aligned across frames;
        2. uses a neighbor list that does not distinguish different atomic types.
        If false, the model
        1. assumes total number of atoms of each atom type aligned across frames;
        2. uses a neighbor list that distinguishes different atomic types.
        """
        return True  # SevenNet应该支持不同原子数的系统
    
    @torch.jit.export
    def has_message_passing(self) -> bool:
        return True
    
    @torch.jit.export
    def get_nnei(self) -> int:
        """Return the total number of selected neighboring atoms in cut-off radius."""
        return self.sel

    @torch.jit.export
    def get_nsel(self) -> int:
        """Return the total number of selected neighboring atoms in cut-off radius."""
        return self.sel
    
    @torch.jit.export
    def model_output_type(self) -> list[str]:
        return ["energy"]
    
    def compute_or_load_stat(self, sampled_func, stat_file_path: Optional[DPPath] = None) -> None:
        """Compute or load statistics parameters."""
        # Determine which keys to compute based on available data
        keys_to_compute = ["energy"]
        #sample_data = sampled_func() if callable(sampled_func) else sampled_func
        #if len(sample_data) > 0 and "force" in sample_data[0]:
        #    keys_to_compute.append("force")
        
        bias_out, std_out = compute_output_stats(
            sampled_func,
            self.get_ntypes(),
            keys=keys_to_compute,
            stat_file_path=stat_file_path,
            rcond=None,
            preset_bias=self.preset_out_bias,
        )
        print("bias_out", bias_out)
        #print("std_out", std_out)
        # Set energy bias (e0) - this is the only bias correction we apply
        if "energy" in bias_out:
            self.e0 = (
                bias_out["energy"]
                .view(self.e0.shape)
                .to(self.e0.dtype)
                .to(self.e0.device)
            )
    
    def serialize(self) -> dict:
        """Serialize the model."""
        return {
            "@class": "Model",
            "@version": 1,
            "type": "matris",
            **self.params,
            "@variables": {
                "e0": to_numpy_array(self.e0),
                **{
                    kk: to_numpy_array(vv)
                    for kk, vv in self.matris_model.state_dict().items()
                },
            },
        }
    
    @classmethod
    def deserialize(cls, data: dict) -> "MatRISModel":
        """Deserialize the model."""
        data = data.copy()
        if not (data.pop("@class") == "Model" and data.pop("type") == "matris"):
            raise ValueError("data is not a serialized MatRISModel")
        
        check_version_compatibility(data.pop("@version"), 1, 1)
        variables = {kk: to_torch_tensor(vv) for kk, vv in data.pop("@variables").items()}
        
        model = cls(**data)
        model.e0 = variables.pop("e0")
        
        if variables:
            model.matris_model.load_state_dict(variables)
        
        return model
    
    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: Optional[list[str]],
        local_jdata: dict,
    ) -> tuple[dict, Optional[float]]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statictics
        type_map : list[str], optional
            The name of each type of atoms
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        float
            The minimum distance between two atoms
        """
        local_jdata_cpy = local_jdata.copy()
        # MatRIS uses pairwise_cutoff instead of r_max --- 这里改了。
        rcut = local_jdata_cpy.get("pairwise_cutoff", local_jdata_cpy.get("r_max", 6.0))
        min_nbor_dist, sel = UpdateSel().update_one_sel(
            train_data,
            type_map,
            rcut,
            local_jdata_cpy["sel"],
            mixed_type=True,
        )
        local_jdata_cpy["sel"] = sel[0]
        return local_jdata_cpy, min_nbor_dist
    
    def model_output_def(self) -> ModelOutputDef:
        """Get output definition."""
        return ModelOutputDef(self.fitting_output_def())
    
    def translated_output_def(self) -> dict[str, Any]:
        """Get translated output definition."""
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": deepcopy(out_def_data["energy"]),
            "energy": deepcopy(out_def_data["energy_redu"]),
        }
        output_def["force"] = deepcopy(out_def_data["energy_derv_r"])
        output_def["force"].squeeze(-2)
        output_def["virial"] = deepcopy(out_def_data["energy_derv_c_redu"])
        output_def["virial"].squeeze(-2)
        output_def["atom_virial"] = deepcopy(out_def_data["energy_derv_c"])
        output_def["atom_virial"].squeeze(-3)
        if "mask" in out_def_data:
            output_def["mask"] = deepcopy(out_def_data["mask"])
        return output_def
    
    @classmethod
    def get_model(cls, model_params: dict) -> "MatRISModel":
        """Get model by parameters."""
        model_params_old = model_params.copy()
        model_params = model_params.copy()
        model_params.pop("type", None)
        
        precision = model_params.pop("precision", "float64")
        if precision == "float32":
            torch.set_default_dtype(torch.float32)
        elif precision == "float64":
            torch.set_default_dtype(torch.float64)
        else:
            raise ValueError(f"precision {precision} not supported")
        

        model = cls(**model_params)
        model.model_def_script = json.dumps(model_params_old)
        return model

    
    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward lower pass - not implemented for native batch approach."""
        return {"energy": torch.zeros(extended_coord.shape[0], 1)}
    
 