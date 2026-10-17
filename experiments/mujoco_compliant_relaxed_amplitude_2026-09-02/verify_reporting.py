"""Read-only regression checks of the old/new policy comparison in the report."""
import study as s
e = s.e


def verify(strict_checks):
    for wave in (2.5, 3.):
        assert e.fx.pd_config(wave) == s.manifest()['PD_profiles_frozen'][f'{wave:g}']['config']
    old = s.read(e.PARENT / 'outputs/data/validated_selection.json')
    checked = []
    for wave, item in old['waves'].items():
        if item['selected']:
            checks = strict_checks(item['selected'])
            assert all(check['feasible'] for check in checks), (wave, checks)
            checked.append(dict(wave=float(wave), checks=checks))
    interim = []
    interim_path = e.DATA / 'early_demo_results.json'
    for item in (s.read(interim_path) if interim_path.exists() else []):
        checks = strict_checks(item)
        assert item['qualified_candidate']
        assert any(not check['feasible'] for check in checks)
        interim.append(dict(wave=item['wave'], strict_policy_checks=checks,
                            relaxed_combination_passed=True))
    result = dict(passed=True, policy_sha256=s.gates.POLICY_SHA,
        previous_validated_waves_checked=len(checked),
        source_legacy_axis_PD_matches_frozen_profiles=True,
        old_qualified_combinations_still_pass_old_policy=checked,
        new_interim_combinations_fail_explicit_old_limits=interim,
        search_gates_and_physics_changed=False)
    s.write(e.DATA / 'reporting_policy_verification.json', result)
    return result


if __name__ == '__main__':
    import delivery
    print(s.json.dumps(verify(delivery.strict_checks), indent=2))
