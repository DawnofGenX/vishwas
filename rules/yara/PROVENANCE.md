# PROVENANCE — YARA rules bundle (`rules/yara/`)

Purpose: enable the `yara_x` evidence family of the `malicious_file` capability
(Finding C, `docs/research/ZERO_RETENTION_E2E_2026-08-21.md`). Loaded via
`VERISAFE_YARA_RULES` (or the in-repo default `src/verisafe/assets/yara_rules/`).

## Source ruleset (fetched exactly once, 2026-08-22)

- **Project:** Mandiant / Google Threat Intelligence — *red_team_tool_countermeasures*
  (YARA/ClamAV/Snort countermeasures for the nation-state red-team toolset
  disclosed by FireEye in Dec 2020; maintained by Mandiant).
- **Source URL:** https://github.com/mandiant/red_team_tool_countermeasures
- **Fetch URL:** https://codeload.github.com/mandiant/red_team_tool_countermeasures/tar.gz/3561b71724dbfa3e2bb78106aaa2d7f8b892c43b
- **Pinned commit:** `3561b71724dbfa3e2bb78106aaa2d7f8b892c43b` (master HEAD at fetch time, committed 2024-03-05T10:19:46Z)
- **Tarball sha256:** `a97c8de0eb9dec508410d2f3a0ae1f76760a94820b4b8a22d7ad1c9101e5eb9a`
- **License:** BSD 2-Clause ("Simplified License") — see upstream `LICENSE.txt`
  (sha256 `a5b8bd9ae96c630bbf229d61c5c5def20274e00be39abfd2fab4d7879ceab462`). Permissive; compatible with this bundle's use.
  Upstream terms: https://github.com/mandiant/red_team_tool_countermeasures/blob/master/LICENSE.txt
- **Why this ruleset:** reputable vendor-published open ruleset, permissive
  license, small (188 KB), high-fidelity production-tier rules designed for
  enterprise deployment (low false-positive rate), and 100% of the per-tool
  files compile under the `yara_x` (YARA-X) engine this project uses — the
  bundled monolith `all-yara.yar` does NOT compile under yara-x, so the
  per-tool files are used instead.

## Composition of this directory

- 79 files `rtc_*.yar` — curated flat subset of upstream `rules/*/production/yara/`
  (1 rule per file, 79 rules). Flat names: `rtc_<TOOL>_<upstream-filename>`.
- `verisafe_starter.yar` — symlink to the in-repo MIT-licensed starter
  (`../../src/verisafe/assets/yara_rules/verisafe_starter.yar`, 3 rules incl.
  `Eicar_Test_File`). Symlinked, not copied, to keep one source of truth.
- Loader constraint honored: `sorted(glob("*.yar*"))[:80]` → exactly 80 entries,
  all verified to compile with `yara_x` from `/home/hermes/pylibs`
  (82 rules loadable total). Bundle size ≈ 109 KB — far below the
  50 MB / 5000-rule subsetting threshold; the only subsetting reason is the
  80-file glob cap.

### Excluded from upstream (and why)

- `all-yara.yar` (repo monolith) — EXCLUDED: fails yara-x compile (`error[E014]: invalid regular expression`; yara-x is stricter than legacy YARA). The per-tool files are the same rules split out; 172 of 172 individual files compile cleanly.
- 77 `production/` files beyond the 79-file cap — EXCLUDED to respect the loader's `glob("*.yar*")[:80]` cap (see `src/verisafe/capabilities/malware_file.py`); selection guarantees at least one rule per tool family (55/55 covered) then fills alphabetically. Full upstream set remains re-fetchable via the pinned tarball.
- 16 non-production-tier files (`supplemental/`, `new/`) — EXCLUDED: lower-fidelity hunting rules; production tier preferred for false-positive safety.

## File manifest (sha256 per file)

| file in this dir | upstream path | sha256 | bytes |
|---|---|---|---|
| rtc_ADPASSHUNT_APT_HackTool_MSIL_ADPassHunt_1.yar | `rules/ADPASSHUNT/production/yara/APT_HackTool_MSIL_ADPassHunt_1.yar` | `156913bdc033add8f761ee096859ba713861b11b4d3bb24fd9be602117e6a43b` | 1258 |
| rtc_ADPASSHUNT_APT_HackTool_MSIL_ADPassHunt_2.yar | `rules/ADPASSHUNT/production/yara/APT_HackTool_MSIL_ADPassHunt_2.yar` | `a960073bbb60b9ea4f6565668683a74a49c5bba15f8fb386c5bf0a5c766c1dcb` | 994 |
| rtc_ADPASSHUNT_CredTheft_MSIL_ADPassHunt_1.yar | `rules/ADPASSHUNT/production/yara/CredTheft_MSIL_ADPassHunt_1.yar` | `f10b43f820eec1f18c49380b4def3ef771356c0b42288ca1b6b1cfe0f536cc56` | 821 |
| rtc_ADPASSHUNT_CredTheft_MSIL_ADPassHunt_2.yar | `rules/ADPASSHUNT/production/yara/CredTheft_MSIL_ADPassHunt_2.yar` | `f2d8de838a50ed723f1d7841e09de3b8525ae0b39876b5e997bad5f6dfe07f8a` | 872 |
| rtc_ALLTHETHINGS_Loader_MSIL_AllTheThings_1.yar | `rules/ALLTHETHINGS/production/yara/Loader_MSIL_AllTheThings_1.yar` | `4260acf844f55bfb98fe405f793f6080213f7234dec76a19eb01cc87991e17d2` | 858 |
| rtc_BASICPIPESHELL_APT_Backdoor_PS1_BASICPIPESHELL_1.yar | `rules/BASICPIPESHELL/production/yara/APT_Backdoor_PS1_BASICPIPESHELL_1.yar` | `ed4e9cf4421b7c1119f9e22219d7af5b50e1d0703120075ddbcb1d3d260ec5a7` | 809 |
| rtc_BELTALOWDA_HackTool_MSIL_SEATBELT_1.yar | `rules/BELTALOWDA/production/yara/HackTool_MSIL_SEATBELT_1.yar` | `a545895c118b62b7442c17c0d8a0cdd84c9a20a426c6e2698f2b013415c6505e` | 1536 |
| rtc_BELTALOWDA_HackTool_MSIL_SEATBELT_2.yar | `rules/BELTALOWDA/production/yara/HackTool_MSIL_SEATBELT_2.yar` | `70b4d1122977a76d01a483a5eb42c82e1fa5c01efbc9162bddb48f11b9477012` | 859 |
| rtc_COREHOUND_HackTool_MSIL_CoreHound_1.yar | `rules/COREHOUND/production/yara/HackTool_MSIL_CoreHound_1.yar` | `8a987eda051b428552ab5a9398136da93bf0f7f01f0ea38b2c189d79f2ace3ac` | 854 |
| rtc_DSHELL_APT_Backdoor_Win_DShell_1.yar | `rules/DSHELL/production/yara/APT_Backdoor_Win_DShell_1.yar` | `24215cf17e19f987f3ab212742cc0c5fd18f3df5945806b8ecc99bd1590c0658` | 5928 |
| rtc_DSHELL_APT_Backdoor_Win_DShell_3.yar | `rules/DSHELL/production/yara/APT_Backdoor_Win_DShell_3.yar` | `4929df58a6b8f5bf03d9ded8f46314f982e42204ba91e83f89d9fc6e645d06c7` | 3093 |
| rtc_DSHELL_APT_Loader_Win32_DShell_1.yar | `rules/DSHELL/production/yara/APT_Loader_Win32_DShell_1.yar` | `673f9803d5c60a87eb8f71a001b2b079cee23fe42662b809ecf3228b037b6f94` | 1011 |
| rtc_DSHELL_APT_Loader_Win32_DShell_2.yar | `rules/DSHELL/production/yara/APT_Loader_Win32_DShell_2.yar` | `9e2e0d2198f9130f357be6c6cd3ffe52d44454413688ac94dfcf3d988fe156ad` | 992 |
| rtc_DSHELL_APT_Loader_Win32_DShell_3.yar | `rules/DSHELL/production/yara/APT_Loader_Win32_DShell_3.yar` | `6e8582451ec83fab46b6dff84f41bd179bc7b4db51107c41c6005d4b3252542f` | 898 |
| rtc_DTRIM_APT_HackTool_MSIL_DTRIM_1.yar | `rules/DTRIM/production/yara/APT_HackTool_MSIL_DTRIM_1.yar` | `592d4b9bb57d3af3c492dd6454dfd7635d82b7948ecd5533d5fa541b260397cb` | 894 |
| rtc_DUEDLLIGENCE_HackTool_MSIL_HOLSTER_1.yar | `rules/DUEDLLIGENCE/production/yara/HackTool_MSIL_HOLSTER_1.yar` | `5078b28c5c40b35a091fb60d42f27720f05fd9c4fd1216f10ab18bedcb76d54f` | 883 |
| rtc_DUEDLLIGENCE_Loader_MSIL_DUEDLLIGENCE_1.yar | `rules/DUEDLLIGENCE/production/yara/Loader_MSIL_DUEDLLIGENCE_1.yar` | `12e894cd25000ee759a9f1def3600faa55f46fb52c17e61d32c4db844da92fee` | 1023 |
| rtc_DUEDLLIGENCE_Loader_MSIL_DUEDLLIGENCE_2.yar | `rules/DUEDLLIGENCE/production/yara/Loader_MSIL_DUEDLLIGENCE_2.yar` | `074a2c0f072cb023cb4c44a6bb7c341d9491fa7f735ef1bb7f4907073be70662` | 596 |
| rtc_DUEDLLIGENCE_Loader_MSIL_DUEDLLIGENCE_3.yar | `rules/DUEDLLIGENCE/production/yara/Loader_MSIL_DUEDLLIGENCE_3.yar` | `fbf6a418cbc9d4e1674cc4a90c7dc0a8dd474c2ef12fe677f0295f732019e85e` | 1231 |
| rtc_DUEDLLIGENCE_MSIL_Launcher_DUEDLLIGENCE_1.yar | `rules/DUEDLLIGENCE/production/yara/MSIL_Launcher_DUEDLLIGENCE_1.yar` | `2cbadee14a6c2ed6e1a6d2b4cde1ac2b0d62fec08d8914ec2b52cad28cf265ce` | 860 |
| rtc_EXCAVATOR_APT_HackTool_Win64_EXCAVATOR_1.yar | `rules/EXCAVATOR/production/yara/APT_HackTool_Win64_EXCAVATOR_1.yar` | `6b740d155919b8c18f146b5bca31740dfe60e1013ae3d5833e2216f1675a2b7a` | 1361 |
| rtc_EXCAVATOR_APT_HackTool_Win64_EXCAVATOR_2.yar | `rules/EXCAVATOR/production/yara/APT_HackTool_Win64_EXCAVATOR_2.yar` | `5f95cd0a5664b69de58a62b3cfad95870c6705e49b0d04d172d030028ea97b09` | 1706 |
| rtc_EXCAVATOR_CredTheft_Win_EXCAVATOR_1.yar | `rules/EXCAVATOR/production/yara/CredTheft_Win_EXCAVATOR_1.yar` | `9add4bc5978efc6131e057b2f313a6bfcfe9ce8e2df426e678c9c7978b236913` | 11182 |
| rtc_EXCAVATOR_CredTheft_Win_EXCAVATOR_2.yar | `rules/EXCAVATOR/production/yara/CredTheft_Win_EXCAVATOR_2.yar` | `3a39490adfa41b70e67223e3b6e2677fe0626d60280f14616c22af4769be2a9c` | 11621 |
| rtc_FLUFFY_APT_HackTool_MSIL_FLUFFY_1.yar | `rules/FLUFFY/production/yara/APT_HackTool_MSIL_FLUFFY_1.yar` | `2b76593a91afdb4acd806de1e67f1b2b3f853aef55d7926fb53840eb52cc08a9` | 1283 |
| rtc_FLUFFY_APT_HackTool_MSIL_FLUFFY_2.yar | `rules/FLUFFY/production/yara/APT_HackTool_MSIL_FLUFFY_2.yar` | `c30a5b92086f0ab338e9d558d5041d3dbc2e45ef7f2bc6c9d43cddd46b5f6dc8` | 825 |
| rtc_G2JS_Builder_MSIL_G2JS_1.yar | `rules/G2JS/production/yara/Builder_MSIL_G2JS_1.yar` | `bbb84e700dfb10ea29fde7809657f579175f711b18370ef43af3f93ef34b5484` | 853 |
| rtc_G2JS_Hunting_B64Engine_DotNetToJScript_Dos.yar | `rules/G2JS/production/yara/Hunting_B64Engine_DotNetToJScript_Dos.yar` | `51fcdaded4ae159b602c910ed6fb4b8ba0aa21f8a91651d87c0fb5ebde5ba935` | 850 |
| rtc_G2JS_Hunting_DotNetToJScript_Functions.yar | `rules/G2JS/production/yara/Hunting_DotNetToJScript_Functions.yar` | `7ce69973f57a07b396d5242598ea3f5d3f2c6d2250f73a0acc17418982c171ec` | 1123 |
| rtc_G2JS_Hunting_GadgetToJScript_1.yar | `rules/G2JS/production/yara/Hunting_GadgetToJScript_1.yar` | `072f3a873838be328fd892e7d9b5fbf8c0792c9bdbc9272ce13322fe5d5f4394` | 769 |
| rtc_GETDOMAINPASSWORDPOLICY_HackTool_MSIL_GETDOMAINPASSWORDPOLICY_1.yar | `rules/GETDOMAINPASSWORDPOLICY/production/yara/HackTool_MSIL_GETDOMAINPASSWORDPOLICY_1.yar` | `3a23e18cc305ba8739bd2553e04723ad88b6308f79a82c90bdb61c817978e162` | 916 |
| rtc_GPOHUNT_APT_HackTool_MSIL_GPOHUNT_1.yar | `rules/GPOHUNT/production/yara/APT_HackTool_MSIL_GPOHUNT_1.yar` | `70db66bb63450d9ba97359f3f6332c1a7766b57964ec4e5f8818af1f59552b7a` | 854 |
| rtc_IMPACKETOBF_Smbexec_HackTool_PY_ImpacketObfuscation_1.yar | `rules/IMPACKETOBF (Smbexec)/production/yara/HackTool_PY_ImpacketObfuscation_1.yar` | `d97576adc1c50c0b54e2238fc48975db4c34775b35b5deadce708c4f00014a3d` | 1084 |
| rtc_IMPACKETOBF_Wmiexec_HackTool_PY_ImpacketObfuscation_2.yar | `rules/IMPACKETOBF (Wmiexec)/production/yara/HackTool_PY_ImpacketObfuscation_2.yar` | `7da094661509b761d725f3f66c57e814de3d91d4717a283fb40364ce65bf2347` | 1080 |
| rtc_INVEIGHZERO_HackTool_MSIL_INVEIGHZERO_1.yar | `rules/INVEIGHZERO/production/yara/HackTool_MSIL_INVEIGHZERO_1.yar` | `37fc162996eb6e2b9c9917fc74dc90b361689f012b8c1b43b4b9ac0ae30e9b20` | 858 |
| rtc_JUSTASK_APT_HackTool_MSIL_JUSTASK_1.yar | `rules/JUSTASK/production/yara/APT_HackTool_MSIL_JUSTASK_1.yar` | `263ea5098398102dd0c58f84320e67e315daf1c13d9580eb7754f483a0f4c3e5` | 854 |
| rtc_KEEFARCE_HackTool_MSIL_KeeFarce_1.yar | `rules/KEEFARCE/production/yara/HackTool_MSIL_KeeFarce_1.yar` | `9aedb3f4b80ffe1dc84fda8d5c904ebfaccbe63834e66ef2367991bb4367aaff` | 852 |
| rtc_KEEPERSIST_HackTool_MSIL_KeePersist_1.yar | `rules/KEEPERSIST/production/yara/HackTool_MSIL_KeePersist_1.yar` | `5923045facacd651af88562d13cab9317d3ccfb8921d25bb34b731f0dfddb519` | 856 |
| rtc_LNKSMASHER_Dropper_LNK_LNKSmasher_1.yar | `rules/LNKSMASHER/production/yara/Dropper_LNK_LNKSmasher_1.yar` | `830d82304d3e1809605ec3b32c4829e6f895b11cc7eafce872566015b49eaeee` | 1022 |
| rtc_LUALOADER_APT_HackTool_MSIL_LUALOADER_1.yar | `rules/LUALOADER/production/yara/APT_HackTool_MSIL_LUALOADER_1.yar` | `69f49767a4457effafbdf1f232110d014af29ecf5b3fbc0bde56af6ab4fe0e10` | 858 |
| rtc_LUALOADER_APT_Loader_MSIL_LUALOADER_1.yar | `rules/LUALOADER/production/yara/APT_Loader_MSIL_LUALOADER_1.yar` | `08badc114fec81194546c84053569505f3b16e52a6a73bed230f41fd1aa2e0ec` | 1187 |
| rtc_LUALOADER_APT_Loader_MSIL_LUALOADER_2.yar | `rules/LUALOADER/production/yara/APT_Loader_MSIL_LUALOADER_2.yar` | `be32ad0baca79d1d3e5ba9a903d9e9c8c7835bff0f7a3b17ff034ddb78312a25` | 1040 |
| rtc_MATRYOSHKA_APT_Builder_PY_MATRYOSHKA_1.yar | `rules/MATRYOSHKA/production/yara/APT_Builder_PY_MATRYOSHKA_1.yar` | `f6b55cec7d48ca14997218f4a53e607206ede0acf0be1a53571fe8cec4ee5c6c` | 920 |
| rtc_MATRYOSHKA_APT_Builder_Win64_MATRYOSHKA_1.yar | `rules/MATRYOSHKA/production/yara/APT_Builder_Win64_MATRYOSHKA_1.yar` | `c16cea926c6258a9ebf949816b726d9b9d53c82d204643eb8d8d10046ad1785a` | 948 |
| rtc_MATRYOSHKA_APT_Dropper_Win64_MATRYOSHKA_1.yar | `rules/MATRYOSHKA/production/yara/APT_Dropper_Win64_MATRYOSHKA_1.yar` | `03001559cbb7d62507d45f628573d26866dfaa01c7f3dbaa9510a3503aeca0c0` | 1354 |
| rtc_MATRYOSHKA_APT_Dropper_Win_MATRYOSHKA_1.yar | `rules/MATRYOSHKA/production/yara/APT_Dropper_Win_MATRYOSHKA_1.yar` | `c97af80474e700342118d329149532e2f46a6ad6d1cb598767653e0b82ed4605` | 872 |
| rtc_MEMCOMP_Loader_MSIL_InMemoryCompilation_1.yar | `rules/MEMCOMP/production/yara/Loader_MSIL_InMemoryCompilation_1.yar` | `1e6595acbadead8b322efb92a72b58a9e73ae8b073dc400e54acf84a6811d415` | 873 |
| rtc_NETASSEMBLYINJECT_Loader_MSIL_NETAssemblyInject_1.yar | `rules/NETASSEMBLYINJECT/production/yara/Loader_MSIL_NETAssemblyInject_1.yar` | `77017b76cf4e07f0dc8433301c578e7ef2024429a049a6146c932d0f6f641213` | 1032 |
| rtc_NETSHSHELLCODERUNNER_Loader_MSIL_NetshShellCodeRunner_1.yar | `rules/NETSHSHELLCODERUNNER/production/yara/Loader_MSIL_NetshShellCodeRunner_1.yar` | `90f41fd8f9c3e157712c468c7987f5c559fb5ac232e11231e36cb9b01b7eec82` | 874 |
| rtc_NOAMCI_APT_HackTool_MSIL_NOAMCI_1.yar | `rules/NOAMCI/production/yara/APT_HackTool_MSIL_NOAMCI_1.yar` | `979d66eb09cacdb76efd86cf53b811397cc6c0052675d40afe55243297ab3f6b` | 933 |
| rtc_PGF_APT_Loader_MSIL_PGF_1.yar | `rules/PGF/production/yara/APT_Loader_MSIL_PGF_1.yar` | `9b5d5929a747405e3824b0907253c51581b2f6d28da146121c0b48c5dbd1578b` | 860 |
| rtc_PREPSHELLCODE_HackTool_MSIL_PrepShellcode_1.yar | `rules/PREPSHELLCODE/production/yara/HackTool_MSIL_PrepShellcode_1.yar` | `4ba4a9b6f02f739703732e9e62c2bae4abcd4316d595fcf23d1619957f2e4eef` | 862 |
| rtc_PUPPYHOUND_HackTool_MSIL_PuppyHound_1.yar | `rules/PUPPYHOUND/production/yara/HackTool_MSIL_PuppyHound_1.yar` | `ae313686b1d956ecc4e0df16445a456868bb23f874cebde804127f13dbe342be` | 1025 |
| rtc_PXELOOT_HackTool_MSIL_PXELOOT_1.yar | `rules/PXELOOT/production/yara/HackTool_MSIL_PXELOOT_1.yar` | `a71c4c3924b1ddd98600e7b9717dc723f0f12a9b0b1437ac0f15f396bb522282` | 855 |
| rtc_REDFLARE_APT_Builder_PY_REDFLARE_1.yar | `rules/REDFLARE/production/yara/APT_Builder_PY_REDFLARE_1.yar` | `458f78b78bd34aa5c192e34f1cebd384d2d46a44477b70522a501a5dbb584cfd` | 859 |
| rtc_REDFLARE_Gorat_APT_Backdoor_MacOS_GORAT_1.yar | `rules/REDFLARE (Gorat)/production/yara/APT_Backdoor_MacOS_GORAT_1.yar` | `ffd98a5398cb3bdf3eb494a2d56d44794435de07a7d4ae04fa52c3fa4cbc864a` | 921 |
| rtc_RESUMEPLEASE_Trojan_Macro_RESUMEPLEASE_1.yar | `rules/RESUMEPLEASE/production/yara/Trojan_Macro_RESUMEPLEASE_1.yar` | `edecf92201cbb576b8ede185f580555553301eea82790bcd40a17aed43e76bbb` | 721 |
| rtc_REVOLVER_APT_HackTool_MSIL_REVOLVER_1.yar | `rules/REVOLVER/production/yara/APT_HackTool_MSIL_REVOLVER_1.yar` | `7103c617f8fb029f3d84aec50df1bbe82c92e25c60d5aac84ee592700c8b2bd0` | 937 |
| rtc_RUBEUS_HackTool_MSIL_Rubeus_1.yar | `rules/RUBEUS/production/yara/HackTool_MSIL_Rubeus_1.yar` | `b0c35b72f1e63d4de688452e857d6cca2d902487757bcd5d5685e5adee678767` | 812 |
| rtc_SAFETYKATZ_HackTool_MSIL_SAFETYKATZ_4.yar | `rules/SAFETYKATZ/production/yara/HackTool_MSIL_SAFETYKATZ_4.yar` | `4d48f4b0d9ffbb4c1f012828330feaee1e6e899f7b39bda2ef27ba71ef60036e` | 863 |
| rtc_SHARPERSIST_HackTool_MSIL_SharPersist_1.yar | `rules/SHARPERSIST/production/yara/HackTool_MSIL_SharPersist_1.yar` | `92eb959ebc2abb7ec0c11f9cdb7b4e603a3cb7beed24ed299f3ddc1ceef65f3f` | 858 |
| rtc_SHARPGENERATOR_Builder_MSIL_SharpGenerator_1.yar | `rules/SHARPGENERATOR/production/yara/Builder_MSIL_SharpGenerator_1.yar` | `51ef5d1e02878363ed6d71a9858e128487ad0c37e718868321c4bdddb7d3900b` | 863 |
| rtc_SHARPIVOT_HackTool_MSIL_SharPivot_1.yar | `rules/SHARPIVOT/production/yara/HackTool_MSIL_SharPivot_1.yar` | `26103751b901e4a752edc83f0760bd06e33db7fe416fe1c46b75acda70122533` | 844 |
| rtc_SHARPPGREP_Tool_MSIL_SharpGrep_1.yar | `rules/SHARPPGREP/production/yara/Tool_MSIL_SharpGrep_1.yar` | `a9c463a5d9d70fb76ca66e6ad85959f7bd329ac31913ef8aad46e2d3e066d04c` | 850 |
| rtc_SHARPSACK_APT_HackTool_MSIL_SHARPSACK_1.yar | `rules/SHARPSACK/production/yara/APT_HackTool_MSIL_SHARPSACK_1.yar` | `1f78d6cb86489989e94b9c3e32431412e3a7e7e5d21e7d0f0a4608ad8177bd47` | 858 |
| rtc_SHARPSCHTASK_HackTool_MSIL_SharpSchtask_1.yar | `rules/SHARPSCHTASK/production/yara/HackTool_MSIL_SharpSchtask_1.yar` | `6bd0de5a02517fa1fb3123bf3485a2779256befada21cfd99b96a068700e8d11` | 860 |
| rtc_SHARPSECTIONINJECTION_Loader_MSIL_CSharpSectionInjection_1.yar | `rules/SHARPSECTIONINJECTION/production/yara/Loader_MSIL_CSharpSectionInjection_1.yar` | `9787a2c5ddf1c3a97244012c6a28e6b47d52314606958e07b974e0282ba1e0ca` | 880 |
| rtc_SHARPSTOMP_APT_HackTool_MSIL_SHARPSTOMP_1.yar | `rules/SHARPSTOMP/production/yara/APT_HackTool_MSIL_SHARPSTOMP_1.yar` | `364915bd950d34233e035df2400342269348e5c9cc1915dc50aabeb85a52dead` | 996 |
| rtc_SHARPUTILS_Tool_MSIL_CSharpUtils_1.yar | `rules/SHARPUTILS/production/yara/Tool_MSIL_CSharpUtils_1.yar` | `e68333267863bfaceecb20506f859208d792dd4fb1d3f323842c82a55a9eaa41` | 1178 |
| rtc_SHARPY_Loader_MSIL_SharPy_1.yar | `rules/SHARPY/production/yara/Loader_MSIL_SharPy_1.yar` | `f7c943876f85551da7d7e589cf55c3bb30720a391a51391806ee5270218394e8` | 846 |
| rtc_SHARPZEROLOGON_HackTool_MSIL_SHARPZEROLOGON_1.yar | `rules/SHARPZEROLOGON/production/yara/HackTool_MSIL_SHARPZEROLOGON_1.yar` | `85cced16d90b21cf398cc76e6e2e7e324eac8a07b73cb91d61ab84614b887eeb` | 871 |
| rtc_SINFULOFFICE_Builder_MSIL_SinfulOffice_1.yar | `rules/SINFULOFFICE/production/yara/Builder_MSIL_SinfulOffice_1.yar` | `1acdc38340e17908ae8c264a875c2afb1fc0b92523030bbaf0d3f25de88a998c` | 859 |
| rtc_TITOSPECIAL_APT_HackTool_MSIL_TITOSPECIAL_1.yar | `rules/TITOSPECIAL/production/yara/APT_HackTool_MSIL_TITOSPECIAL_1.yar` | `893c810066e9eb4334c03a3f35114a20f5462797702edf89896a66101a18fd61` | 1083 |
| rtc_TRIMBISHOP_APT_Loader_MSIL_TRIMBISHOP_1.yar | `rules/TRIMBISHOP/production/yara/APT_Loader_MSIL_TRIMBISHOP_1.yar` | `acf3e69f79711379595071fbacfcb05e5b3490241d6354e241d478c3e9a35384` | 1417 |
| rtc_UNCATEGORIZED_APT_HackTool_MSIL_DNSOVERHTTPS_C2_1.yar | `rules/UNCATEGORIZED/production/yara/APT_HackTool_MSIL_DNSOVERHTTPS_C2_1.yar` | `139e96e45a777cee885e3fe505f57a61c6d916e856eef865646100b03c0bfb76` | 960 |
| rtc_WILDCHILD_APT_Loader_MSIL_WILDCHILD_1.yar | `rules/WILDCHILD/production/yara/APT_Loader_MSIL_WILDCHILD_1.yar` | `44329c864cb4ccd400bf6088d2cb7fd02b1dee960dec450465780908b2da8e08` | 1212 |
| rtc_WMIRUNNER_Loader_MSIL_WMIRunner_1.yar | `rules/WMIRUNNER/production/yara/Loader_MSIL_WMIRunner_1.yar` | `9dc42d914efb59c27f513dee384e705cbad322f42929a7a5abe8d065ee0ea13e` | 852 |
| rtc_WMISHARP_HackTool_MSIL_WMISharp_1.yar | `rules/WMISHARP/production/yara/HackTool_MSIL_WMISharp_1.yar` | `3685b8fb17a36fd8a80e2763f7e89b15cb0a9ad707e856cd349d1089df4d903e` | 852 |
| rtc_WMISPY_APT_HackTool_MSIL_WMISPY_2.yar | `rules/WMISPY/production/yara/APT_HackTool_MSIL_WMISPY_2.yar` | `dc0e8f39f9a33755764e7107ba49898c68d6a365715919e348ee85d226b750aa` | 1018 |

## Durable enablement (env wiring)

`VERISAFE_YARA_RULES=/home/hermes/verisafe/rules/yara`

- The live systemd user unit `~/.config/systemd/user/verisafe-webhook.service`
  (owned by the main agent — do not edit the unit) already loads
  `EnvironmentFile=/home/hermes/verisafe/deploy/verisafe-secrets.env`, so the
  line above was added to that file. After a `systemctl --user restart
  verisafe-webhook.service` the webhook path picks the bundle up.
- CLI / manual runs: `export VERISAFE_YARA_RULES=/home/hermes/verisafe/rules/yara`
  before `python3 -m verisafe.app cli --file ... --media-type document`.
- Without the env var the loader falls back to the in-repo starter dir
  (`src/verisafe/assets/yara_rules/`, EICAR + 2 generic rules) — degraded but
  functional.

## Update procedure

Re-fetch the pinned tarball (or a newer pinned commit) exactly as above,
re-run the compile check (`yara_x.compile` per file, drop failures), refresh
this manifest, and keep total files ≤ 79 (+ starter symlink).
