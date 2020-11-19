import os, pytest, glob


@pytest.mark.parametrize("app", glob.glob('examples/*.json', recursive=False))
def test_all_apps(app):
    os.system('''python dse_graph_analysis.py -f ''' + app + ''' 0''')

    assert(os.path.exists("outputs"))
    assert(os.path.exists("pdf"))
    assert(os.path.exists("outputs/peak_eqs"))
    assert(os.path.exists("outputs/rewrite_rules"))
    assert(os.path.exists("outputs/verilog"))
    assert(os.path.exists("outputs/peak_eqs/peak_eq_0.py"))
    assert(os.path.exists("outputs/PE.json"))
    assert(os.path.exists("outputs/rewrite_rules/rewrite_rule_0.json"))
    assert(os.path.exists("outputs/verilog/PE.json"))
    assert(os.path.exists("outputs/verilog/PE.v"))
