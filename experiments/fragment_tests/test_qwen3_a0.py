"""Tiny real Qwen3 architecture: exact cuts, reference parity and corruption gates."""
import json
from pathlib import Path
import pytest
import torch
from vibeweights.abi import ContractError
from vibeweights.qwen3_fragments import extract, FragmentStore, compare, _hash, _write
from transformers import Qwen3Config,Qwen3ForCausalLM


@pytest.fixture
def snapshot(tmp_path):
    root=tmp_path/'source';root.mkdir()
    torch.manual_seed(19)
    config=Qwen3Config(vocab_size=128,hidden_size=32,intermediate_size=64,
        num_hidden_layers=5,num_attention_heads=4,num_key_value_heads=2,
        head_dim=8,max_position_embeddings=128,tie_word_embeddings=True,
        use_sliding_window=False,attention_bias=False)
    model=Qwen3ForCausalLM(config).eval()
    model.save_pretrained(root,safe_serialization=True)
    (root/'LICENSE').write_text('Synthetic test fixture; no external weights.\n',encoding='utf-8')
    return root


def build(snapshot,root,scales=(1,2,3)):
    return extract(snapshot,root,model_id='test/qwen3-synthetic',revision='0'*40,
                   origin='SYNTHETIC',scales=scales)


def test_multiscale_reference_parity_and_source_free_replay(snapshot,tmp_path):
    root=tmp_path/'fragments';m=build(snapshot,root)
    assert m['partitions']=={'1':[[0,1],[1,2],[2,3],[3,4],[4,5]],
                            '2':[[0,2],[2,4],[4,5]],'3':[[0,3],[3,5]]}
    assert len(m['blocks'])==5 and len(list(root.glob('*.safetensors')))==7
    assert m['logical_parameters']==sum(p.numel() for p in Qwen3ForCausalLM(Qwen3Config.from_pretrained(snapshot)).parameters())
    cases=[[7],[12,17,4],[2,3,5,7,11,13,17]]
    store=FragmentStore(root)
    report=compare(snapshot,store,cases)
    assert report['status']=='PASS'
    assert all(len(rows)==5 for rows in report['partitions'].values())
    expected=store.replay(1,cases)
    # The original full model is inaccessible during independent replay.
    snapshot.rename(tmp_path/'unavailable-source')
    store=FragmentStore(root)
    for scale in (2,3):
        for a,b in zip(store.replay(scale,cases),expected):assert torch.equal(a,b)
    span=store.load_span(2,4)
    h=torch.zeros(1,3,32)
    assert span.forward(h).shape==h.shape
    assert all(not p.requires_grad for layer in span.layers for p in layer.parameters())
    with pytest.raises(ContractError,match='latent boundary'):span.forward(torch.zeros(1,3,31))


def test_immutable_artifact_and_config_integrity(snapshot,tmp_path):
    root=tmp_path/'fragments';m=build(snapshot,root)
    with pytest.raises(ContractError,match='already exists'):build(snapshot,root)
    store=FragmentStore(root)
    path=root/(m['blocks'][0]['sha256']+'.safetensors')
    raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
    with pytest.raises(ContractError,match='artifact integrity'):store.load_span(0,1)
    cfg=root/'config.json';cfg.write_bytes(cfg.read_bytes()+b' ')
    with pytest.raises(ContractError,match='config/license integrity'):FragmentStore(root)


def test_partition_and_source_binding(snapshot,tmp_path):
    root=tmp_path/'fragments';m=build(snapshot,root)
    with pytest.raises(ContractError,match='unknown fragment scale'):FragmentStore(root).replay(4,[[1]])
    with pytest.raises(ContractError,match='fragment tokens'):FragmentStore(root).replay(2,[[128]])
    with pytest.raises(ContractError,match='fragment span'):FragmentStore(root).load_span(2,6)
    with pytest.raises(ContractError,match='reference token bounds'):compare(snapshot,FragmentStore(root),[[1]*128])
    license_file=snapshot/'LICENSE';license_file.write_bytes(b'changed')
    with pytest.raises(ContractError,match='reference source differs'):compare(snapshot,FragmentStore(root),[[1]])
    m['partitions']['2']=[[0,2],[3,5]]
    _write(root/'manifest.json',{'manifest':m,'sha256':_hash(m)})
    with pytest.raises(ContractError,match='partition coverage'):FragmentStore(root)


def test_reject_unsupported_sources_and_no_partial_output(snapshot,tmp_path):
    root=tmp_path/'fragments'
    with pytest.raises(ContractError,match='fragment scale'):build(snapshot,root,scales=(6,))
    assert not root.exists()
    cfg=snapshot/'config.json';data=json.loads(cfg.read_text(encoding='utf-8'))
    data['tie_word_embeddings']=False;_write(cfg,data)
    with pytest.raises(ContractError,match='unsupported Qwen3'):build(snapshot,root)
    assert not root.exists()
    data['tie_word_embeddings']=True;_write(cfg,data)
    with pytest.raises(ContractError,match='within budget'):
        extract(snapshot,root,model_id='test',revision='0'*40,origin='SYNTHETIC',scales=[1],byte_budget=1)
    assert not root.exists()
