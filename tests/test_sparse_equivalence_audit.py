"""Old-vs-gather-first semantic checks on valid finite model states."""
import copy
import pytest
import torch
from axosim.interfaces import AxoSimPopulation
from axosim.support_surrogate import AdaptiveSupportP4Surrogate, SupportP4Config


def original_sparse(population, summary, contact, values, *, time_steps, external=None):
    efficacy = population.synaptic_efficacies(external).reshape(-1).index_select(0, contact)
    routes = population._contact_route_features().reshape(-1, population.neuron.config.route_feature_dim).index_select(0, contact)
    contributions = routes.to(values.dtype) * (values * efficacy).to(values.dtype).unsqueeze(-1)
    sums = torch.index_add(values.new_zeros(population.population_size * time_steps,
                                          population.neuron.config.route_feature_dim),0,summary,contributions)
    return population.neuron.forward_runtime_coefficients(
        sums.reshape(population.population_size,time_steps,-1),runtime_coefficients=population._runtime_coefficients())


def make_population(device, dtype, contacts=7):
    torch.manual_seed(198)
    cfg = SupportP4Config(input_dim=9,route_feature_dim=5,token_dim=4,state_dim=3,
                          morphology_ids=('m0','m1'),behavior_adaptation=True)
    pop = AxoSimPopulation(AdaptiveSupportP4Surrogate(cfg,torch.randn(2,9,5)),
        morphology_indices=torch.tensor([0,1,0]),
        contact_branch_indices=torch.arange(3*contacts).reshape(3,contacts)%9).to(device=device,dtype=dtype)
    with torch.no_grad():
        pop.synaptic_log_efficacy.normal_(std=.2)
        pop.behavior_adapter.normal_(std=.02)
    pop.neuron.route_features.requires_grad_(True)
    return pop


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float64,torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('case',['duplicates','empty','zero_contacts','external'])
def test_original_and_gather_first_outputs_and_all_gradients(device,dtype,case):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    contacts=0 if case=='zero_contacts' else 7
    old=make_population(device,dtype,contacts)
    new=copy.deepcopy(old)
    empty=case in ('empty','zero_contacts')
    summary=torch.tensor([] if empty else [0,0,3,25,31,46,70],dtype=torch.int32,device=device)
    contact=torch.tensor([] if empty else [1,1,3,10,9,7,17],dtype=torch.int64,device=device)
    x=torch.tensor([] if empty else [1.,.5,-.25,2.,-.4,1.,.2],dtype=dtype,device=device,requires_grad=True)
    y=x.detach().clone().requires_grad_(True)
    ext_old=old.synaptic_log_efficacy.detach().clone().requires_grad_(True) if case=='external' else None
    ext_new=ext_old.detach().clone().requires_grad_(True) if ext_old is not None else None
    a=original_sparse(old,summary,contact,x,time_steps=24,external=ext_old)
    b=new.forward_sparse_contacts(summary,contact,y,time_steps=24,synaptic_log_efficacy=ext_new)
    tolerance={torch.float64:1e-11,torch.float32:2e-6,torch.float16:2e-3,torch.bfloat16:2e-2}[dtype]
    torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance)
    a.square().sum().backward();b.square().sum().backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=tolerance,atol=tolerance)
    for po,pn in zip(old.parameters(),new.parameters(),strict=True):
        assert (po.grad is None)==(pn.grad is None)
        if po.grad is not None:
            torch.testing.assert_close(po.grad,pn.grad,rtol=tolerance,atol=tolerance)
    torch.testing.assert_close(old.neuron.route_features.grad,new.neuron.route_features.grad,rtol=tolerance,atol=tolerance)
    if ext_old is not None:
        torch.testing.assert_close(ext_old.grad,ext_new.grad,rtol=tolerance,atol=tolerance)
        assert old.synaptic_log_efficacy.grad is None
        assert new.synaptic_log_efficacy.grad is None


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_sparse_index_dtypes_and_out_of_range_validation(device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pop=make_population(device,torch.float32)
    values=torch.ones(2,device=device)
    for summary_dtype in (torch.int32,torch.int64):
        for contact_dtype in (torch.int32,torch.int64):
            output=pop.forward_sparse_contacts(torch.tensor([0,24],device=device,dtype=summary_dtype),
                torch.tensor([0,7],device=device,dtype=contact_dtype),values,time_steps=24)
            assert torch.isfinite(output).all()
    with pytest.raises(ValueError,match='same target neuron'):
        pop.forward_sparse_contacts(torch.tensor([0],device=device),torch.tensor([7],device=device),values[:1],time_steps=24)
    with pytest.raises(ValueError,match='outside'):
        pop.forward_sparse_contacts(torch.tensor([0],device=device),torch.tensor([21],device=device),values[:1],time_steps=24)
