import torch

from axosim.axomamba import AxoMambaConfig, AxoPyTorchMamba, default_axomamba_config
from axosim.checkpoint import load_checkpoint, save_checkpoint
from axosim.mamba_official import BranchOfficialMambaConfig, BranchPyTorchMamba
from axosim.mamba_pytorch import SimpleMambaModel


def test_simple_mamba_trunk_parameters_receive_gradients():
    torch.manual_seed(0)
    model = SimpleMambaModel(input_dim=8, d_model=4, n_layers=1, d_state=2, d_conv=2, expand=2)
    x = torch.randn(2, 5, 8)
    target = torch.randn(2, 5, 2)

    loss = (model(x) - target).square().mean()
    loss.backward()

    required = [
        "layers.0.A_log",
        "layers.0.D",
        "layers.0.conv1d.weight",
        "layers.0.x_proj.weight",
        "layers.0.dt_proj.weight",
    ]
    grads = dict(model.named_parameters())
    for name in required:
        grad = grads[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name


def test_simple_mamba_checkpoint_round_trips(tmp_path):
    torch.manual_seed(0)
    model = SimpleMambaModel(input_dim=8, d_model=4, n_layers=1, d_state=2, d_conv=2, expand=2)
    path = tmp_path / "mamba.pt"

    save_checkpoint(model, path, metadata={"epoch": 1})
    loaded, metadata = load_checkpoint(path)

    assert isinstance(loaded, SimpleMambaModel)
    assert loaded.config == model.config
    assert metadata["model_kind"] == "mamba-pytorch"
    assert metadata["epoch"] == 1


def test_branch_pytorch_mamba_trunk_parameters_receive_gradients():
    torch.manual_seed(0)
    config = BranchOfficialMambaConfig(
        num_input=8,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=4,
        input_to_synapse_routing="neuronio_routing",
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        branch_gain=True,
        branch_nonlinearity="tanh",
        soma_multiplicative_gate=True,
        integrative_mixer=True,
        integrative_mixer_short_kernel=2,
        integrative_mixer_long_kernel=3,
    )
    model = BranchPyTorchMamba(config)
    x = torch.randn(2, 5, 8)
    target = torch.randn(2, 5, 2)

    loss = (model(x) - target).square().mean()
    loss.backward()

    grads = dict(model.named_parameters())
    required = [
        "blocks.0.mamba.A_log",
        "blocks.0.mamba.D",
        "blocks.0.mamba.conv1d.weight",
        "blocks.0.mamba.x_proj.weight",
        "blocks.0.mamba.dt_proj.weight",
        "branch_projection.weight",
    ]
    for name in required:
        grad = grads[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name


def test_branch_pytorch_mamba_checkpoint_round_trips(tmp_path):
    config = BranchOfficialMambaConfig(
        num_input=8,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=4,
        input_to_synapse_routing="neuronio_routing",
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
    )
    model = BranchPyTorchMamba(config)
    path = tmp_path / "branch_mamba.pt"

    save_checkpoint(model, path, metadata={"epoch": 2})
    loaded, metadata = load_checkpoint(path)

    assert isinstance(loaded, BranchPyTorchMamba)
    assert loaded.config == model.config
    assert metadata["model_kind"] == "branch-mamba-pytorch"
    assert metadata["epoch"] == 2


def test_axomamba_default_config_promotes_best_compact_stack():
    config = default_axomamba_config(num_input=8, num_branch=2, num_synapse_per_branch=4, model_dim=4)

    assert isinstance(config, AxoMambaConfig)
    assert config.block_type == "mamba1"
    assert config.branch_gain is True
    assert config.branch_nonlinearity == "tanh"
    assert config.integrative_mixer is True
    assert config.integrative_mixer_position == "pre"
    assert config.soma_multiplicative_gate is True


def test_axomamba_pytorch_checkpoint_round_trips(tmp_path):
    torch.manual_seed(0)
    config = default_axomamba_config(
        num_input=8,
        num_branch=2,
        num_synapse_per_branch=4,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        integrative_mixer_short_kernel=2,
        integrative_mixer_long_kernel=3,
    )
    model = AxoPyTorchMamba(config)
    x = torch.randn(2, 5, 8)
    output = model(x)
    assert output.shape == (2, 5, 2)

    path = tmp_path / "axomamba.pt"
    save_checkpoint(model, path, metadata={"epoch": 3})
    loaded, metadata = load_checkpoint(path)

    assert loaded.__class__.__name__ == "AxoPyTorchMamba"
    assert loaded.config == model.config
    assert metadata["model_kind"] == "axomamba-pytorch"
    assert metadata["epoch"] == 3
