import pytest
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


def test_axomamba_default_config_promotes_ablation_retained_stack():
    config = default_axomamba_config(num_input=8, num_branch=2, num_synapse_per_branch=4, model_dim=4)

    assert isinstance(config, AxoMambaConfig)
    assert config.block_type == "mamba1"
    assert config.synapse_gain_scale is None
    assert config.branch_gain is False
    assert config.branch_nonlinearity == "tanh"
    assert config.integrative_mixer is False
    assert config.integrative_mixer_position == "pre"
    assert config.separate_heads is True
    assert config.soma_multiplicative_gate is False
    assert config.spike_voltage_coupling_scale == 4.0

    promoted = default_axomamba_config()
    assert (promoted.num_branch, promoted.num_synapse_per_branch) == (118, 11)
    assert (promoted.model_dim, promoted.state_dim) == (58, 23)
    assert promoted.branch_bilinear_rank == 0
    assert promoted.branch_bilinear_scale is None
    assert promoted.synapse_gain_scale is None


def test_spike_voltage_coupling_uses_soma_without_adding_parameters():
    common = dict(
        num_input=8,
        num_branch=2,
        num_synapse_per_branch=4,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        soma_multiplicative_gate=True,
    )
    torch.manual_seed(11)
    control = AxoPyTorchMamba(AxoMambaConfig(**common))
    torch.manual_seed(11)
    coupled = AxoPyTorchMamba(AxoMambaConfig(**common, spike_voltage_coupling_scale=1.5))
    coupled.load_state_dict(control.state_dict())
    x = torch.randn(2, 5, 8)

    control_output = control(x)
    coupled_output = coupled(x)

    assert sum(parameter.numel() for parameter in coupled.parameters()) == sum(
        parameter.numel() for parameter in control.parameters()
    )
    torch.testing.assert_close(coupled_output[..., 1], control_output[..., 1])
    torch.testing.assert_close(
        coupled_output[..., 0],
        control_output[..., 0] + 1.5 * control_output[..., 1],
    )

def test_morphology_embedding_scale_zero_is_noop_and_context_is_required():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=4,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
    )
    torch.manual_seed(41)
    control = AxoPyTorchMamba(AxoMambaConfig(**common))
    torch.manual_seed(41)
    inactive = AxoPyTorchMamba(
        AxoMambaConfig(
            **common,
            morphology_ids=["specimen_a", "specimen_b"],
            morphology_embedding_scale=0.0,
        )
    )
    x = torch.randn(2, 7, 8)
    morphology_indices = torch.tensor([0, 1])

    torch.testing.assert_close(inactive(x, morphology_indices=morphology_indices), control(x))
    assert sum(parameter.numel() for parameter in inactive.parameters()) == (
        sum(parameter.numel() for parameter in control.parameters()) + 8
    )
    with pytest.raises(ValueError, match="morphology_indices"):
        inactive(x)

    active = AxoPyTorchMamba(
        AxoMambaConfig(
            **common,
            morphology_ids=["specimen_a", "specimen_b"],
            morphology_embedding_scale=1.0,
        )
    )
    active.load_state_dict(inactive.state_dict())
    with torch.no_grad():
        assert active.morphology_embedding is not None
        active.morphology_embedding.weight[0].copy_(torch.tensor([-0.5, 0.0, 0.5, 1.0]))
        active.morphology_embedding.weight[1].copy_(torch.tensor([1.0, 0.5, 0.0, -0.5]))
    assert not torch.allclose(
        active(x, morphology_indices=torch.tensor([0, 0])),
        active(x, morphology_indices=torch.tensor([1, 1])),
    )


def test_morphology_embedding_can_be_removed_while_other_conditioning_remains():
    config = AxoMambaConfig(
        num_input=8,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=4,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        morphology_ids=["specimen_a", "specimen_b"],
        morphology_embedding_scale=None,
        morphology_synapse_gain_scale=1.0,
    )
    model = AxoPyTorchMamba(config)
    x = torch.randn(2, 7, 8)

    assert model.morphology_embedding is None
    output = model(x, morphology_indices=torch.tensor([0, 1]))
    assert output.shape == (2, 7, 2)
    with pytest.raises(ValueError, match="morphology_indices"):
        model(x)


def test_branch_bilinear_interaction_is_matched_and_zero_initialized():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=4,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        branch_bilinear_rank=2,
    )
    torch.manual_seed(67)
    inactive = AxoPyTorchMamba(AxoMambaConfig(**common, branch_bilinear_scale=0.0))
    torch.manual_seed(67)
    active = AxoPyTorchMamba(AxoMambaConfig(**common, branch_bilinear_scale=1.0))
    active.load_state_dict(inactive.state_dict())
    branch_inputs = torch.randn(2, 7, 2)
    hidden = torch.randn(2, 7, 4)

    assert sum(parameter.numel() for parameter in active.parameters()) == sum(
        parameter.numel() for parameter in inactive.parameters()
    )
    torch.testing.assert_close(active._apply_branch_bilinear(hidden, branch_inputs), hidden)
    assert active.branch_bilinear_out is not None
    loss = active._apply_branch_bilinear(hidden, branch_inputs).square().mean()
    loss.backward()
    assert active.branch_bilinear_out.weight.grad is not None
    assert active.branch_bilinear_out.weight.grad.abs().sum() > 0

    with torch.no_grad():
        active.branch_bilinear_out.weight.fill_(0.25)
    assert not torch.allclose(active._apply_branch_bilinear(hidden, branch_inputs), hidden)


def test_synapse_gains_are_identity_initialized_and_exactly_controlled():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=4,
        num_synapse_per_branch=2,
        model_dim=6,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
    )
    torch.manual_seed(103)
    control = AxoPyTorchMamba(AxoMambaConfig(**common, synapse_gain_scale=0.0))
    torch.manual_seed(103)
    active = AxoPyTorchMamba(AxoMambaConfig(**common, synapse_gain_scale=1.0))
    active.load_state_dict(control.state_dict())
    x = torch.randn(2, 9, 8)

    assert sum(parameter.numel() for parameter in active.parameters()) == sum(
        parameter.numel() for parameter in control.parameters()
    )
    torch.testing.assert_close(active.synapse_log_gain, torch.zeros(4, 2))
    torch.testing.assert_close(active(x), control(x))

    active(x).square().mean().backward()
    control(x).square().mean().backward()
    assert active.synapse_log_gain.grad is not None
    assert active.synapse_log_gain.grad.abs().sum() > 0
    assert control.synapse_log_gain.grad is not None
    torch.testing.assert_close(
        control.synapse_log_gain.grad,
        torch.zeros_like(control.synapse_log_gain),
    )

    with torch.no_grad():
        active.synapse_log_gain[0, 0] = 1.0
    assert not torch.allclose(active._branch_reduce(x), control._branch_reduce(x))


def test_synapse_gain_scale_must_be_non_negative():
    with pytest.raises(ValueError, match="synapse_gain_scale"):
        AxoPyTorchMamba(
            AxoMambaConfig(
                num_input=8,
                num_output=2,
                num_branch=4,
                num_synapse_per_branch=2,
                model_dim=4,
                num_layers=1,
                state_dim=2,
                conv_kernel=2,
                expansion=2,
                synapse_gain_scale=-1.0,
            )
        )


def test_morphology_synapse_gains_are_exactly_controlled_and_specific():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=4,
        num_synapse_per_branch=2,
        model_dim=6,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        synapse_gain_scale=1.0,
        morphology_ids=["morph-a", "morph-b"],
        morphology_embedding_scale=0.1,
    )
    torch.manual_seed(103)
    control = AxoPyTorchMamba(
        AxoMambaConfig(**common, morphology_synapse_gain_scale=0.0)
    )
    torch.manual_seed(103)
    active = AxoPyTorchMamba(
        AxoMambaConfig(**common, morphology_synapse_gain_scale=1.0)
    )
    active.load_state_dict(control.state_dict())
    x = torch.randn(2, 11, 8)
    morphology_indices = torch.tensor([0, 1])

    assert active.morphology_synapse_log_gain.shape == (2, 4, 2)
    assert sum(parameter.numel() for parameter in active.parameters()) == sum(
        parameter.numel() for parameter in control.parameters()
    )
    torch.testing.assert_close(
        active(x, morphology_indices=morphology_indices),
        control(x, morphology_indices=morphology_indices),
    )

    active(x, morphology_indices=morphology_indices).square().mean().backward()
    control(x, morphology_indices=morphology_indices).square().mean().backward()
    assert active.morphology_synapse_log_gain.grad is not None
    assert active.morphology_synapse_log_gain.grad.abs().sum() > 0
    torch.testing.assert_close(
        control.morphology_synapse_log_gain.grad,
        torch.zeros_like(control.morphology_synapse_log_gain),
    )

    with torch.no_grad():
        active.morphology_synapse_log_gain[0, 0, 0] = 1.0
    assert not torch.allclose(
        active(x, morphology_indices=morphology_indices),
        control(x, morphology_indices=morphology_indices),
    )


def test_morphology_synapse_gain_configuration_is_strict():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=4,
        num_synapse_per_branch=2,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
    )
    with pytest.raises(ValueError, match="morphology_synapse_gain_scale"):
        AxoPyTorchMamba(AxoMambaConfig(**common, morphology_synapse_gain_scale=1.0))
    with pytest.raises(ValueError, match="morphology_synapse_gain_scale"):
        AxoPyTorchMamba(
            AxoMambaConfig(
                **common,
                morphology_ids=["morph-a"],
                morphology_embedding_scale=0.1,
                morphology_synapse_gain_scale=-1.0,
            )
        )


def test_morphology_coordinate_synapse_operator_is_exactly_controlled():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=4,
        num_synapse_per_branch=2,
        model_dim=6,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        morphology_ids=["morph-a", "morph-b"],
        morphology_embedding_scale=0.1,
        morphology_synapse_gain_scale=1.0,
        morphology_synapse_feature_dim=3,
        morphology_synapse_feature_hidden=4,
    )
    torch.manual_seed(107)
    control = AxoPyTorchMamba(
        AxoMambaConfig(**common, morphology_synapse_feature_scale=0.0)
    )
    torch.manual_seed(107)
    active = AxoPyTorchMamba(
        AxoMambaConfig(**common, morphology_synapse_feature_scale=1.0)
    )
    active.load_state_dict(control.state_dict())
    features = torch.arange(2 * 8 * 3, dtype=torch.float32).view(2, 8, 3) / 48.0
    control.morphology_synapse_features.copy_(features)
    active.morphology_synapse_features.copy_(features)
    x = torch.randn(2, 11, 8)
    morphology_indices = torch.tensor([0, 1])

    assert active.morphology_synapse_features.shape == (2, 8, 3)
    assert sum(parameter.numel() for parameter in active.parameters()) == sum(
        parameter.numel() for parameter in control.parameters()
    )
    torch.testing.assert_close(
        active(x, morphology_indices=morphology_indices),
        control(x, morphology_indices=morphology_indices),
    )

    active(x, morphology_indices=morphology_indices).square().mean().backward()
    control(x, morphology_indices=morphology_indices).square().mean().backward()
    assert active.morphology_synapse_feature_out.weight.grad is not None
    assert active.morphology_synapse_feature_out.weight.grad.abs().sum() > 0
    torch.testing.assert_close(
        control.morphology_synapse_feature_out.weight.grad,
        torch.zeros_like(control.morphology_synapse_feature_out.weight),
    )

    with torch.no_grad():
        active.morphology_synapse_feature_out.weight.fill_(0.1)
    assert not torch.allclose(
        active(x, morphology_indices=morphology_indices),
        control(x, morphology_indices=morphology_indices),
    )


def test_morphology_coordinate_synapse_configuration_is_strict():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=4,
        num_synapse_per_branch=2,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
    )
    with pytest.raises(ValueError, match="morphology_synapse_feature"):
        AxoPyTorchMamba(
            AxoMambaConfig(
                **common,
                morphology_synapse_feature_dim=3,
                morphology_synapse_feature_hidden=4,
                morphology_synapse_feature_scale=1.0,
            )
        )
    with pytest.raises(ValueError, match="morphology_synapse_feature"):
        AxoPyTorchMamba(
            AxoMambaConfig(
                **common,
                morphology_ids=["morph-a"],
                morphology_embedding_scale=0.1,
                morphology_synapse_feature_dim=3,
                morphology_synapse_feature_hidden=0,
                morphology_synapse_feature_scale=1.0,
            )
        )


def test_axomamba_structural_sharing_reuses_modules_after_checkpoint_load(tmp_path):
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=4,
        model_dim=8,
        num_layers=2,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        local_tcn_scale=1.0,
    )
    control = AxoPyTorchMamba(AxoMambaConfig(**common))
    shared = AxoPyTorchMamba(
        AxoMambaConfig(
            **common,
            share_local_tcn_pointwise=True,
        )
    )

    assert shared.local_tcn_blocks[0].pointwise is shared.local_tcn_blocks[1].pointwise
    assert shared.local_tcn_blocks[1].pointwise is shared.local_tcn_blocks[2].pointwise
    assert sum(parameter.numel() for parameter in shared.parameters()) < sum(
        parameter.numel() for parameter in control.parameters()
    )

    path = tmp_path / "shared.pt"
    save_checkpoint(shared, path)
    loaded, _ = load_checkpoint(path)
    assert loaded.local_tcn_blocks[0].pointwise is loaded.local_tcn_blocks[2].pointwise


def test_local_tcn_operator_is_controlled_multiscale_and_causal():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=4,
        num_synapse_per_branch=2,
        model_dim=6,
        num_layers=2,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
    )
    torch.manual_seed(125)
    control = AxoPyTorchMamba(
        AxoMambaConfig(**common, local_tcn_scale=0.0)
    )
    torch.manual_seed(125)
    dilated = AxoPyTorchMamba(
        AxoMambaConfig(**common, local_tcn_scale=1.0)
    )
    torch.manual_seed(125)
    undilated = AxoPyTorchMamba(
        AxoMambaConfig(
            **common,
            local_tcn_scale=1.0,
            local_tcn_mode="undilated",
        )
    )
    torch.manual_seed(125)
    interlayer = AxoPyTorchMamba(
        AxoMambaConfig(
            **common,
            local_tcn_scale=1.0,
            local_tcn_position="interlayer",
        )
    )
    torch.manual_seed(125)
    post = AxoPyTorchMamba(
        AxoMambaConfig(
            **common,
            local_tcn_scale=1.0,
            local_tcn_position="post",
        )
    )
    dilated.load_state_dict(control.state_dict())
    undilated.load_state_dict(control.state_dict())
    interlayer.load_state_dict(control.state_dict())
    post.load_state_dict(control.state_dict())
    x = torch.randn(2, 41, 8)

    assert sum(parameter.numel() for parameter in dilated.parameters()) == sum(
        parameter.numel() for parameter in control.parameters()
    )
    torch.testing.assert_close(dilated(x), control(x))
    torch.testing.assert_close(undilated(x), control(x))
    torch.testing.assert_close(interlayer(x), control(x))
    torch.testing.assert_close(post(x), control(x))

    dilated(x).square().mean().backward()
    control(x).square().mean().backward()
    assert dilated.local_tcn_out.weight.grad is not None
    assert dilated.local_tcn_out.weight.grad.abs().sum() > 0
    torch.testing.assert_close(
        control.local_tcn_out.weight.grad,
        torch.zeros_like(control.local_tcn_out.weight),
    )

    with torch.no_grad():
        pattern = torch.arange(
            dilated.local_tcn_out.weight.numel(),
            dtype=dilated.local_tcn_out.weight.dtype,
        ).view_as(dilated.local_tcn_out.weight)
        dilated.local_tcn_out.weight.copy_(0.01 * pattern)
        undilated.local_tcn_out.weight.copy_(dilated.local_tcn_out.weight)
        interlayer.local_tcn_out.weight.copy_(dilated.local_tcn_out.weight)
        post.local_tcn_out.weight.copy_(dilated.local_tcn_out.weight)
    assert not torch.allclose(dilated(x), undilated(x))
    assert not torch.allclose(dilated(x), interlayer(x))
    assert not torch.allclose(dilated(x), post(x))

    changed = x.clone()
    changed[:, 30:] = torch.randn_like(changed[:, 30:])
    torch.testing.assert_close(dilated(x)[:, :30], dilated(changed)[:, :30])


def test_soma_tcn_correction_can_isolate_spike_coupling_and_is_causal():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=4,
        num_synapse_per_branch=2,
        model_dim=6,
        num_layers=2,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        separate_heads=True,
        spike_voltage_coupling_scale=4.0,
    )
    models = {}
    for name, scale, coupling_source in (
        ("control", 0.0, "corrected"),
        ("corrected_coupling", 1.0, "corrected"),
        ("base_coupling", 1.0, "base"),
    ):
        torch.manual_seed(132)
        models[name] = AxoPyTorchMamba(
            AxoMambaConfig(
                **common,
                soma_tcn_scale=scale,
                soma_tcn_coupling_source=coupling_source,
            )
        )
    state = models["control"].state_dict()
    for model in models.values():
        model.load_state_dict(state)
    x = torch.randn(2, 41, 8)

    initial = {name: model(x) for name, model in models.items()}
    for output in initial.values():
        torch.testing.assert_close(output, initial["control"])
    assert len({sum(p.numel() for p in model.parameters()) for model in models.values()}) == 1

    with torch.no_grad():
        pattern = torch.arange(
            models["control"].soma_tcn_out.weight.numel(),
            dtype=models["control"].soma_tcn_out.weight.dtype,
        ).view_as(models["control"].soma_tcn_out.weight)
        for model in models.values():
            model.soma_tcn_out.weight.copy_(0.01 * pattern)
    outputs = {name: model(x) for name, model in models.items()}
    torch.testing.assert_close(outputs["control"], initial["control"])
    torch.testing.assert_close(
        outputs["corrected_coupling"][..., 1], outputs["base_coupling"][..., 1]
    )
    assert not torch.allclose(
        outputs["corrected_coupling"][..., 0], outputs["base_coupling"][..., 0]
    )
    torch.testing.assert_close(
        outputs["base_coupling"][..., 0], initial["control"][..., 0]
    )

    changed = x.clone()
    changed[:, 30:] = torch.randn_like(changed[:, 30:])
    torch.testing.assert_close(
        models["base_coupling"](x)[:, :30],
        models["base_coupling"](changed)[:, :30],
    )


def test_sparse_branch_subunits_are_exactly_controlled_and_reachable():
    common = dict(
        num_input=12,
        num_output=2,
        num_branch=3,
        num_synapse_per_branch=4,
        model_dim=6,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        synapse_gain_scale=1.0,
        branch_nonlinearity="tanh",
        branch_nonlinearity_scale_init=0.1,
        branch_subunits=2,
    )
    torch.manual_seed(113)
    promoted = AxoPyTorchMamba(
        AxoMambaConfig(**{key: value for key, value in common.items() if key != "branch_subunits"})
    )
    torch.manual_seed(113)
    dense = AxoPyTorchMamba(AxoMambaConfig(**common, branch_subunit_routing="dense"))
    torch.manual_seed(113)
    sparse = AxoPyTorchMamba(AxoMambaConfig(**common, branch_subunit_routing="contiguous"))
    torch.manual_seed(113)
    inactive = AxoPyTorchMamba(
        AxoMambaConfig(
            **common,
            branch_subunit_routing="contiguous",
            branch_subunit_scale=0.0,
        )
    )
    x = torch.randn(2, 11, 12)

    assert dense.branch_subunit_projection.in_features == 6
    assert sum(parameter.numel() for parameter in dense.parameters()) == sum(
        parameter.numel() for parameter in sparse.parameters()
    )
    torch.testing.assert_close(
        dense.branch_subunit_mask.square().sum(dim=-1),
        sparse.branch_subunit_mask.square().sum(dim=-1),
    )
    torch.testing.assert_close(dense(x), promoted(x))
    torch.testing.assert_close(sparse(x), promoted(x))
    torch.testing.assert_close(inactive(x), promoted(x))

    sparse(x).square().mean().backward()
    assert sparse.branch_subunit_projection.weight.grad is not None
    assert sparse.branch_subunit_projection.weight.grad.abs().sum() > 0
    inactive(x).square().mean().backward()
    torch.testing.assert_close(
        inactive.branch_subunit_projection.weight.grad,
        torch.zeros_like(inactive.branch_subunit_projection.weight),
    )

    with torch.no_grad():
        sparse.branch_subunit_projection.weight[0].fill_(0.01)
        inactive.branch_subunit_projection.weight[0].fill_(0.01)
    assert not torch.allclose(sparse(x), promoted(x))
    torch.testing.assert_close(inactive(x), promoted(x))


def test_branch_subunit_configuration_is_strict():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=4,
        num_synapse_per_branch=2,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
    )
    with pytest.raises(ValueError, match="branch_subunits"):
        AxoPyTorchMamba(AxoMambaConfig(**common, branch_subunits=-1))
    with pytest.raises(ValueError, match="branch_subunit_routing"):
        AxoPyTorchMamba(
            AxoMambaConfig(**common, branch_subunits=2, branch_subunit_routing="invalid")
        )
    with pytest.raises(ValueError, match="branch_subunit_scale"):
        AxoPyTorchMamba(AxoMambaConfig(**common, branch_subunits=2, branch_subunit_scale=-1.0))


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
    output = model(x, morphology_indices=torch.tensor([0, 1]))
    assert output.shape == (2, 5, 2)

    path = tmp_path / "axomamba.pt"
    save_checkpoint(model, path, metadata={"epoch": 3})
    loaded, metadata = load_checkpoint(path)

    assert loaded.__class__.__name__ == "AxoPyTorchMamba"
    assert loaded.config == model.config
    assert metadata["model_kind"] == "axomamba-pytorch"
    assert metadata["epoch"] == 3


def test_axomamba_checkpoint_load_ignores_retired_experiment_config(tmp_path):
    model = AxoPyTorchMamba(
        default_axomamba_config(
            num_input=8,
            num_branch=2,
            num_synapse_per_branch=4,
            model_dim=4,
            num_layers=1,
            state_dim=2,
            conv_kernel=2,
        )
    )
    path = tmp_path / "legacy-axomamba.pt"
    save_checkpoint(model, path)
    payload = torch.load(path, weights_only=False)
    payload["config"].update(
        {
            "final_norm_mix": 0.0,
            "hybrid_current_kernel": 257,
            "hybrid_current_mode": None,
            "input_load_conditioning_scale": None,
            "liquid_soma_kernel": 129,
            "liquid_soma_mode": None,
            "liquid_soma_taus": None,
            "soma_final_norm_mix": 0.0,
            "soma_spike_waveform_detach": False,
            "soma_spike_waveform_kernel": 0,
            "soma_spike_waveform_scale": None,
            "spike_coupling_soma_source": None,
            "morphology_film_scale": None,
            "morphology_branch_gain_scale": None,
            "morphology_reinject_scale": None,
            "polarity_group_port": None,
            "branch_local_mamba_scale": None,
            "branch_local_model_dim": 1,
            "branch_local_state_dim": 25,
            "branch_memory_taus": None,
            "branch_memory_kernel": 257,
            "branch_memory_scale": None,
            "parameter_padding": 0,
            "recurrent_tail": None,
            "morphology_soma_affine_scale": None,
            "synapse_trace_kernel": 129,
            "synapse_trace_mixture": "heterogeneous",
            "synapse_trace_residual": False,
            "synapse_trace_taus": None,
        }
    )
    payload["state_dict"].pop("branch_subunit_mask")
    torch.save(payload, path)

    loaded, _ = load_checkpoint(path)

    assert loaded.__class__.__name__ == "AxoPyTorchMamba"
    assert loaded.config == model.config
