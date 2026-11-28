"""Causal scanner: known toy effects plus actual tiny Qwen3 execution."""
from types import SimpleNamespace
import pytest
import torch
from vibeweights.abi import ContractError
from vibeweights.competence_mapping import regions,observe,bypass,scan
from transformers import Qwen3Config,Qwen3ForCausalLM


class Add(torch.nn.Module):
    def __init__(self,value):
        super().__init__();self.value=value
    def forward(self,hidden_states):return hidden_states+self.value


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=torch.nn.Module()
        self.model.layers=torch.nn.ModuleList([Add(2),Add(-1),Add(0)])
        self.config=SimpleNamespace(vocab_size=4,max_position_embeddings=8)
    def forward(self,input_ids,use_cache=False):
        x=torch.zeros(1,input_ids.shape[1],1)
        for layer in self.model.layers:x=layer(hidden_states=x)
        logits=torch.cat((x,-x,torch.zeros_like(x),torch.zeros_like(x)),dim=-1)
        return SimpleNamespace(logits=logits)


def cases():
    return [{'id':'a','domain':'math','family':'x','tokens':[0],'correct':0,'foil':1},
            {'id':'b','domain':'math','family':'x','tokens':[1],'correct':1,'foil':0}]


def test_known_causal_contribution_and_baseline_failure_exclusion():
    model=Toy().eval().requires_grad_(False)
    r=scan(model,cases(),scales=[2,3])
    domain=r['domains']['math']
    assert domain['baseline_preferred']==1
    rows={(x['start'],x['stop']):x for x in domain['regions']}
    assert rows[0,1]['lost_preference']==1
    assert rows[0,1]['mean_margin_drop']==4
    assert rows[1,2]['mean_margin_drop']==-2
    assert rows[2,3]['mean_margin_drop']==0
    assert model(input_ids=torch.tensor([[0]])).logits[0,-1,0]==1
    assert len(regions(28,[6,8,16]))==39


def test_hook_cleanup_and_invalid_inputs():
    model=Toy().eval()
    layer=model.model.layers[0]
    with pytest.raises(RuntimeError):
        with bypass(model.model.layers,0,1):raise RuntimeError('interrupted')
    assert not layer._forward_hooks
    with pytest.raises(RuntimeError):
        with observe(model.model.layers,{}):raise RuntimeError('interrupted')
    assert not layer._forward_hooks
    with pytest.raises(ContractError):bypass(model.model.layers,-1,2).__enter__()
    with pytest.raises(ContractError):scan(model,[*cases(),cases()[0]],scales=[2])


def test_real_tiny_qwen3_all_spans():
    torch.manual_seed(19)
    config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=1,
        head_dim=8,max_position_embeddings=32,tie_word_embeddings=True,
        use_sliding_window=False,attention_bias=False)
    config._attn_implementation='eager'
    model=Qwen3ForCausalLM(config).eval().requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    report=scan(model,cases(),scales=[2,3])
    assert len(report['cases'])==2
    assert len(report['regions'])==5
    assert all(len(x['activations'])==3 and len(x['bypass'])==5 for x in report['cases'])
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all(not layer._forward_hooks for layer in model.model.layers)
