"""Argument check for the MatRIS models."""

from dargs import Argument
from deepmd.utils.argcheck import model_args_plugin


@model_args_plugin.register("matris")
def matris_model_args() -> Argument:
    """Arguments for the MatRIS model."""
    return _matris_model_args_common("matris")


def _matris_model_args_common(model_name: str) -> Argument:
    """Arguments for the MatRIS model."""
    doc_sel = "Maximum number of neighbor atoms"
    doc_num_layers = "Number of interaction layers"
    doc_node_feat_dim = "Dimension of node (atom) features"
    doc_edge_feat_dim = "Dimension of edge (two-body) features"
    doc_three_body_feat_dim = "Dimension of three-body (angle) features"
    doc_mlp_hidden_dims = "Hidden dimensions for MLP layers in readout head"
    doc_dropout = "Dropout rate in MLP layers"
    doc_use_bias = "Whether to use bias in linear layers"
    doc_distance_expansion = "Type of distance expansion basis function (Bessel or Gaussian)"
    doc_three_body_expansion = "Type of three-body expansion (fourier or SH)"
    doc_num_radial = "Number of radial basis functions for two-body interactions"
    doc_num_angular = "Number of angular basis functions for three-body interactions"
    doc_max_l = "Maximum l value for spherical harmonics (SH expansion)"
    doc_max_n = "Maximum n value for spherical harmonics (SH expansion)"
    doc_envelope_exponent = "Polynomial envelope exponent for smooth cutoff"
    doc_graph_conv_mlp = "Type of MLP in graph convolution (mlp or gatemlp)"
    doc_activation_type = "Activation function type (silu, relu, etc.)"
    doc_norm_type = "Normalization type (layer, batch, rms, etc.)"
    doc_pairwise_cutoff = "Cutoff radius for atom graph (two-body, in Angstrom)"
    doc_three_body_cutoff = "Cutoff radius for line graph (three-body, in Angstrom)"
    doc_use_smoothed_for_delta_edge = "Whether to use smoothed features for edge refinement"
    doc_learnable_basis = "Whether basis function parameters are learnable"
    doc_is_intensive = "Whether output energy is per-atom (True) or total (False)"
    doc_is_conservation = "Whether to use conservative (gradient-based) force/stress"
    doc_reference_energy = "Reference energy dataset name (e.g., 'mptrj', None for no reference)"
    
    return Argument(
        model_name,
        dict,
        [
            Argument("sel", [int, str], optional=False, doc=doc_sel),
            Argument("num_layers", int, optional=True, default=10, doc=doc_num_layers),
            Argument("node_feat_dim", int, optional=True, default=128, doc=doc_node_feat_dim),
            Argument("edge_feat_dim", int, optional=True, default=128, doc=doc_edge_feat_dim),
            Argument("three_body_feat_dim", int, optional=True, default=128, doc=doc_three_body_feat_dim),
            Argument("mlp_hidden_dims", list[int], optional=True, default=[128, 256, 128], doc=doc_mlp_hidden_dims),
            Argument("dropout", float, optional=True, default=0.0, doc=doc_dropout),
            Argument("use_bias", bool, optional=True, default=False, doc=doc_use_bias),
            Argument("distance_expansion", str, optional=True, default="Bessel", doc=doc_distance_expansion),
            Argument("three_body_expansion", str, optional=True, default="fourier", doc=doc_three_body_expansion),
            Argument("num_radial", int, optional=True, default=7, doc=doc_num_radial),
            Argument("num_angular", int, optional=True, default=7, doc=doc_num_angular),
            Argument("max_l", int, optional=True, default=4, doc=doc_max_l),
            Argument("max_n", int, optional=True, default=4, doc=doc_max_n),
            Argument("envelope_exponent", int, optional=True, default=8, doc=doc_envelope_exponent),
            Argument("graph_conv_mlp", str, optional=True, default="gatemlp", doc=doc_graph_conv_mlp),
            Argument("activation_type", str, optional=True, default="silu", doc=doc_activation_type),
            Argument("norm_type", str, optional=True, default="layer", doc=doc_norm_type),
            Argument("pairwise_cutoff", float, optional=True, default=6.0, doc=doc_pairwise_cutoff),
            Argument("three_body_cutoff", float, optional=True, default=4.5, doc=doc_three_body_cutoff),
            Argument("use_smoothed_for_delta_edge", bool, optional=True, default=True, doc=doc_use_smoothed_for_delta_edge),
            Argument("learnable_basis", bool, optional=True, default=True, doc=doc_learnable_basis),
            Argument("is_intensive", bool, optional=True, default=True, doc=doc_is_intensive),
            Argument("is_conservation", bool, optional=True, default=True, doc=doc_is_conservation),
            Argument("reference_energy", [str, type(None)], optional=True, default=None, doc=doc_reference_energy),
        ],
        doc=f"{model_name.replace('_', ' ').title()} model for materials simulation",
    )