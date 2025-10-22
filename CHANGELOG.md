## [1.1.5](https://github.com/RussellSB/pytrendy/compare/v1.1.4...v1.1.5) (2025-10-22)


### Bug Fixes

* **abrupt shaving:** Fix issue of infinite loop as it breaks down abrupt segment into abrupt subsegments.  ([#14](https://github.com/RussellSB/pytrendy/issues/14)) ([f3da256](https://github.com/RussellSB/pytrendy/commit/f3da256a9ea4fca626be41921dcad7623f26671d))

## [1.1.4](https://github.com/RussellSB/pytrendy/compare/v1.1.3...v1.1.4) (2025-10-19)


### Bug Fixes

* **noise detection:** Major revamp. Trend detection now much less sensitive to noise spikes. ([#13](https://github.com/RussellSB/pytrendy/issues/13)) ([f66a636](https://github.com/RussellSB/pytrendy/commit/f66a636ad109d968936914429ae66bb3dea07076))

## [1.1.3](https://github.com/RussellSB/pytrendy/compare/v1.1.2...v1.1.3) (2025-10-16)


### Bug Fixes

* **noise detection:** Handle spike segments on gradual trends much better. ([#12](https://github.com/RussellSB/pytrendy/issues/12)) ([9b8e204](https://github.com/RussellSB/pytrendy/commit/9b8e204148903a707f1183b175f20dce8232174f))

## [1.1.2](https://github.com/RussellSB/pytrendy/compare/v1.1.1...v1.1.2) (2025-10-15)


### Bug Fixes

* resolving merge conflicts & ensuring relative imports work correctly with __init__ for subpackages ([c1c2491](https://github.com/RussellSB/pytrendy/commit/c1c24913e13ca607ea0036344e56e6af7139e92b))

## [1.1.1](https://github.com/RussellSB/pytrendy/compare/v1.1.0...v1.1.1) (2025-10-15)


### Bug Fixes

* **deployment:** checking out again to try handle edge case of deployment failure and rerun on version bump. ([d137ccb](https://github.com/RussellSB/pytrendy/commit/d137ccbf9372a963fc95201d3f1d9fc4a87918bf))

# [1.1.0](https://github.com/RussellSB/pytrendy/compare/v1.0.8...v1.1.0) (2025-10-15)


### Bug Fixes

* **abrupt detection:** don't expand abrupt detection, keep precise. Necessary when order of execution of refine is changed. ([6052be9](https://github.com/RussellSB/pytrendy/commit/6052be919bcbd6531e0e39494a5b1f4fe3041f09))
* **abrupt detection:** handling for more special edge case scenarios, where focal abrupt not in centre but on right ([1fe6e16](https://github.com/RussellSB/pytrendy/commit/1fe6e162111d3bb069b55c5303d876603c4473d3))
* **abrupt detection:** improved brown fix bug logic. Make it more legible, and generalisable. ([5509178](https://github.com/RussellSB/pytrendy/commit/55091780956f08e8ade7d16960d620d386812913))
* **abrupt detection:** shaving abrupt segments out, even when multiple detected under the same rolling window. Improves precision, and does not leave large enough abrupt changes behind. ([580463c](https://github.com/RussellSB/pytrendy/commit/580463c4665698948bcd110d9cf59830d9514dd5))
* **abrupt shave:** fixing is not prev trend logic. Make it still functional for 4 spikes test case with spike at 2025-02-25 ([c66ed2e](https://github.com/RussellSB/pytrendy/commit/c66ed2ec41dacaeb62719e86b36f0d9aca016713))
* **abrupt shave:** make so second pass doesnt cover unnessesarry steps when same segment has not been re-classified. Also handled for edge case in new noisy 10 scenario crash. ([a7cc99e](https://github.com/RussellSB/pytrendy/commit/a7cc99efea42395331c18210b8fa9c269abbae9c))
* **abrupt shaving:** improved abrupt subseg logic so that it doesnt try to force abrupt subseg at end when not connected to going down. This was a rare edge case that happened in light of noise ([36c2732](https://github.com/RussellSB/pytrendy/commit/36c2732be498e11f1f9877e05727df616afa9ec6))
* abrupt sub-segmentation. FIx when it tries go out of bounds. Have rail guards to just look at previous day, otherwise min/max df index if out of bounds ([839123b](https://github.com/RussellSB/pytrendy/commit/839123b297b0de0dcdac33d09b7a0cef9b70b9af))
* **abrupt trend detection:** making more robust to edge case of neighbouring abrupt segments. Will still precisely measure from centre and not confuse with neighbours. ([33b82da](https://github.com/RussellSB/pytrendy/commit/33b82dadd64379a318ed9ec53c2450d3df115322))
* adjusting threshold flat to be <= . Allows it to enable flatline 0 scenarios ([1cf0e3e](https://github.com/RussellSB/pytrendy/commit/1cf0e3efa6a38467421a9fe7f1d9290927b335ff))
* **brown bug edge case:** green would get moved over red causing brown. This is because abrupts would be shaved to abrupt spikes regardless of direction, stacking over each other. Now enforced sensitivity to direciton. Makes sure it aligns before it moves. ([b0d1690](https://github.com/RussellSB/pytrendy/commit/b0d1690c3b0f9de2514e1282ea9f4b2208f80014))
* catering for edge case of no segments detected. Only spotted during dev of clean_segments. Should not occur for time series ([e057bca](https://github.com/RussellSB/pytrendy/commit/e057bca961c65ab57419566f961d5c41fff02ee7))
* **change rank:** improving change rank to not tag out all segments, only directional ones ([c5622b7](https://github.com/RussellSB/pytrendy/commit/c5622b7e2a77e23642f205c3b81accd4ce9e3254))
* **class signals:** making abrupt modelslightly longer so more robust in comparison to gradual ([2555171](https://github.com/RussellSB/pytrendy/commit/25551713373cd8801a94448ee54fb36d99301ded))
* **clean artifacts:** handle for partial overlap, so that when noise accidentally makes abrupt classify as gradual, it can be reclassified as abrupt and appropriately catered for. ([d0e570d](https://github.com/RussellSB/pytrendy/commit/d0e570dd95c46f351215f2ce6cc6f7a321159da0))
* **clean artifacts:** improving clean artifacts with has_overlap_prev as well as next. Helps noise detection, when false positive gradual trends gets moved to small periods over noise. Filter them out ([f3fba86](https://github.com/RussellSB/pytrendy/commit/f3fba863b4316624598c5f6b0a3752cf5bbbcde7))
* **clean artifacts:** improving overlap conditions to cater for more edge cases. When trend overlaps noise. And when overlap is abrupt (actually would want shorter not longer). ([bae5a02](https://github.com/RussellSB/pytrendy/commit/bae5a02190929c1485693713a2d7b7efca3fb512))
* **clean artifacts:** include in has inverse logic, if detected as trend incorrectly when shifting over, and now is actually flat. Remove this case. ([2301bc3](https://github.com/RussellSB/pytrendy/commit/2301bc357d8259057d58f7b6fdea50dfa6b26659))
* **clean artifacts:** making logic generalise for abrupt case with unaccounted for scenarios. Was showing an overlap with z_score thresh line 194 at  0.6 instead of 1. Still settled best to leave at 1. Even if it makes it insensitive to minor abrupt at start for synth 1 test case. ([ab6bc5a](https://github.com/RussellSB/pytrendy/commit/ab6bc5a8d6aa5368dba50a0d92bb100db6143c1f))
* **data loader:** making windows compatible, with not just linux specific seperators '/' ([d04c49c](https://github.com/RussellSB/pytrendy/commit/d04c49c715cb93e52d6c43ac912ce531028f7188))
* **data loader:** Path object instead of str ([cc24e4d](https://github.com/RussellSB/pytrendy/commit/cc24e4d1d4bfa80a413da30467c1a0204bdf1958))
* detect trends robust to string columns in wide dataframes, filters to value col at start ([a7c286d](https://github.com/RussellSB/pytrendy/commit/a7c286dd46609ee15639cd180b35e4c3935bbb83))
* dont move noise if a large spike outlier ([9064b1a](https://github.com/RussellSB/pytrendy/commit/9064b1a520f514308bc319f68ac5a34d953421b5))
* **edge case:** no trends case crashing because it tries to access change_rank when doesnt exist ([41c15ff](https://github.com/RussellSB/pytrendy/commit/41c15ffc62c561baa172ed7064ec656ce055258a))
* **expand-contract:** cater for edge case of when all values are flat. Make sure to choose latest for start and earliest for end in this case. ([6d7eda3](https://github.com/RussellSB/pytrendy/commit/6d7eda30e6d3d2e80e73fe060293b545e975acce))
* **flat detection:** can only detect flat when there isnt a noise spike. ([3802705](https://github.com/RussellSB/pytrendy/commit/38027059d53afea52650abe4e437512ef4c75fa3))
* **flat detection:** improved so not sensitive to mostly 0 signals ([ebe6578](https://github.com/RussellSB/pytrendy/commit/ebe6578f18fae3dbb7cf9eb798f75e960f824d1b))
* **flat detection:** setting minimum length for flat in get segment to 0, so it can be grouped up with sporadic points around it. Improves edge case of noise spikes for flats. ([d871df1](https://github.com/RussellSB/pytrendy/commit/d871df184f51de35b52af16bfc9615b2510c831e))
* **flat:** reverting initial smooth window for std rolling, so that its more sensitive and precise to small flat segments ([914fe7a](https://github.com/RussellSB/pytrendy/commit/914fe7abcb5ca23583b4a6f04132ed11814be7cf))
* **gradual:** make the swallowing more robust to gradual test case. It will be able to stretch between niehgbouring segment adjustments, and exchanged flexibly (flats between up and down) ([5f6e2c9](https://github.com/RussellSB/pytrendy/commit/5f6e2c92bd197cbe00583dc7cce7039fc544cd03))
* **grouping:** changed behaviour now so that it groups abrupts when theyre exactly touching each other consequietively. ([50d2f52](https://github.com/RussellSB/pytrendy/commit/50d2f52a1f94b6a7e7c5edd64da302a79a59aab4))
* **grouping:** dont group up abrupt cases with each other ([3a8e66e](https://github.com/RussellSB/pytrendy/commit/3a8e66eb3fd01492d4bbd45c8b54b5fa22048e97))
* improving clean artifacts. Catering for edge case of same-direction overlaps. This now comes when abrupt is set to not be adjusted as a neighbour during expand contract logic. In this case its just touching a gradual, in light of minor noise that seperates the two. ([64651be](https://github.com/RussellSB/pytrendy/commit/64651be28ef1678079f2728c4dda19a8df326a66))
* **next-prev-shifts:** updated to cater for "swallowing" edge cases. When neighbours completely get overlapped with expansion. Also started debugging other edge case ([0caf3f8](https://github.com/RussellSB/pytrendy/commit/0caf3f8cbe369f81ed4aad964c2f08f019d67049))
* **noise detection:** bring back distinguish logic, except rely on a simple start vs end check to see if of similar magnitudes, rather than rely on complex DTW classification so early in the code ([b2ff121](https://github.com/RussellSB/pytrendy/commit/b2ff121e627483fe7763966900afeec98ccbaca7))
* **noise detection:** cater for edge case of very abrupt noise outlier. Stretch it out slightly so it is still visible enough to persist. ([e35f2f5](https://github.com/RussellSB/pytrendy/commit/e35f2f5e6f07c8198f3efc497fdde4dcc7887f5e))
* **noise detection:** catering for out of bounds pad when checking if abrupt or noisy distinction ([ad32f3b](https://github.com/RussellSB/pytrendy/commit/ad32f3b86d9a899c7d41f83d154f8793589420fd))
* **noise detection:** improving it so that its more robust to leading zero signals before abrupt uptrend. ([129a701](https://github.com/RussellSB/pytrendy/commit/129a7014ade7b993d6796cbcd857e62aab875f1f))
* **noise detection:** making it so it still increases on noise levels. Made the DTW first pass on an exact segment fit, then try on a padding. ([4487f1a](https://github.com/RussellSB/pytrendy/commit/4487f1aa52b07de0fb44ab68155640612a3a5ac8))
* **noise detection:** making noise detection distinguish between noise and abrupt changes with DTW. If abrupt change signal, ignores, and lets get detected by smoothed_deriv ([99a267c](https://github.com/RussellSB/pytrendy/commit/99a267c64ed19af0bbedc7203679260877ab4d4e))
* **noise detection:** making noise detection more precise to spike-type noise segments. Also catering for new edge cases, so still detects noise under gaussian noise test case as it should. ([527fb24](https://github.com/RussellSB/pytrendy/commit/527fb2420d746d3f027706ee2940072037c2a860))
* **padding:** improving padding logic so generalises to new abrupt and next seg logic. rephrasing gradual to be first, so it refers to next gradual segment with accuracy when padding ([36dfe0f](https://github.com/RussellSB/pytrendy/commit/36dfe0fbcd75a38192b29ee0a22cf516e97dbe83))
* **plot:** also padding flat for case of spikey-type noise, as would be done for abrupt case ([6384b0a](https://github.com/RussellSB/pytrendy/commit/6384b0ab3bf7df04b6efd8734272e56fdb9a5d84))
* **refine:** improving group up logic to apply to gradual. Also better generalising subseg logic for shave abrupt so doesnt crash when subseg end too far from start ([8ef6f42](https://github.com/RussellSB/pytrendy/commit/8ef6f42cff18cf08dafa7af9628758866401446b))
* removing noise detection distinguish logic, to improve edge case of when noise outliers are present. The abrupt breakdown logic handles for cases it doesnt exactly match anyways. ([bef5b8f](https://github.com/RussellSB/pytrendy/commit/bef5b8f139cf6156ee792ee223500144500bc902))
* **results pytrendy:** handling no segments detected in results object ([a3ac41f](https://github.com/RussellSB/pytrendy/commit/a3ac41fb9fb113aec7646cc359be9109f61133e2))
* **tweak:** making flat & noise more sensitive to pickup. Better than showing white strips. ([4febda0](https://github.com/RussellSB/pytrendy/commit/4febda0b669627d84a1a9656e259b399998a8864))
* **visual:** abrupt trend leaving white line when neighbouring ([23417d3](https://github.com/RussellSB/pytrendy/commit/23417d34f8eeaa954a52e1e548e842a25c9e097a))
* **visual:** no left displacement for abrupt trends ([5fe2136](https://github.com/RussellSB/pytrendy/commit/5fe21369137528a54dc178cccc21283190961ba0))
* **visual:** visualise segment separation when of same direction type & neighbouring. Mostly useful for when grouping toggled off ([6ed9a80](https://github.com/RussellSB/pytrendy/commit/6ed9a8062edcc25b266aeb1066b7bd0d96c28230))
* **warnings:** fixing warnings in analyse segments so doesnt divide by zero ([0e4ee15](https://github.com/RussellSB/pytrendy/commit/0e4ee1516aa72f0985f04fafe297e36875971909))


### Features

* adding extra functionality to has_inverse() in clean_artifacts. Checks that total change is also consistent to direction, if not it cleans it outs ([37411fe](https://github.com/RussellSB/pytrendy/commit/37411fe1dfc56d3ccfbae7c557029024400e6a2b))
* **fill in flats:** filling in flats in gaps between segments. Assumes coverage of post-processing covers all edge cases, and remaining white gaps should all be flat. ([5c33bbd](https://github.com/RussellSB/pytrendy/commit/5c33bbd734aa9dde03590e638f85988d565c952b))
* improving interface of results. Now results.df instead of results.segments_df. ANd can also call summarised alternative via simply results.df_summary instead of results.summary['df'] ([6866c3b](https://github.com/RussellSB/pytrendy/commit/6866c3bef27a8c26b4a32855260e415e9762edb5))
* Major revamp on core signal_processing and post_processing logic. Much more robust to edge cases now. ([6c53790](https://github.com/RussellSB/pytrendy/commit/6c53790823198034dd2f3f4f9cd18dd44a243235)), closes [#8](https://github.com/RussellSB/pytrendy/issues/8)

## [1.0.8](https://github.com/RussellSB/pytrendy/compare/v1.0.7...v1.0.8) (2025-09-02)


### Bug Fixes

* **edge case:** no trends case crashing because it tries to access change_rank when doesnt exist ([45ff56c](https://github.com/RussellSB/pytrendy/commit/45ff56cc9e6e2815de7946dd5c6e6f28d07a542c))
* **visual:** abrupt plot, making it visually tighter while not affecting presentation choice for gradual and other signals ([f175c62](https://github.com/RussellSB/pytrendy/commit/f175c62c41a3e13cea31ef8343147abb0c251363))
* **visual:** abrupt trend leaving white line when neighbouring ([86fd02a](https://github.com/RussellSB/pytrendy/commit/86fd02a266bc3db1e001735cb0abe0e2f90aaf91))
* **visual:** no left displacement for abrupt trends ([0f33e08](https://github.com/RussellSB/pytrendy/commit/0f33e0891cd32e5782ba6bc24310024a7d56d12b))

## [1.0.7](https://github.com/RussellSB/pytrendy/compare/v1.0.6...v1.0.7) (2025-09-01)


### Bug Fixes

* **ci:** remove semantic release dry run mode, to allow version bumping ([29e3b69](https://github.com/RussellSB/pytrendy/commit/29e3b69895300523e35f0863b7e1cbc5f2f593d2))
* **ci:** resolving non-version bumped workflow issue ([f1ccf50](https://github.com/RussellSB/pytrendy/commit/f1ccf50fe930589024b3ea4e82af249ea36ebebd))
* **ci:** run semantic release trigger ([92dcc80](https://github.com/RussellSB/pytrendy/commit/92dcc802063138c76c96a47e3fca12cdf3002bd8))
* python 3.10 bug of single quotation marks in f string ([88e35dc](https://github.com/RussellSB/pytrendy/commit/88e35dc2cb74c72688d3e17ed8c0ebc264fed930))

## [1.0.6](https://github.com/RussellSB/pytrendy/compare/v1.0.5...v1.0.6) (2025-09-01)


### Bug Fixes

* **docs:** updating image links in README to raw github to work on pypi ([d8c0733](https://github.com/RussellSB/pytrendy/commit/d8c0733c0b960d0d76883b9ad3bd19738f3a373c))

## [1.0.5](https://github.com/RussellSB/pytrendy/compare/v1.0.4...v1.0.5) (2025-09-01)


### Bug Fixes

* **docs:** plot display on PyPi ([609c112](https://github.com/RussellSB/pytrendy/commit/609c11255b49f8bb4c5ae293da35d85f39e4cac7))

## [1.0.4](https://github.com/RussellSB/pytrendy/compare/v1.0.3...v1.0.4) (2025-09-01)


### Bug Fixes

* **docs:** update README.md to be more clear and concise. ([0acd8e3](https://github.com/RussellSB/pytrendy/commit/0acd8e3dc5ae84d34260a1303b28834d0df6ca90))

## [1.0.3](https://github.com/RussellSB/pytrendy/compare/v1.0.2...v1.0.3) (2025-09-01)


### Bug Fixes

* **ci:** triggering version patch bump to update docs on pypi ([35beed4](https://github.com/RussellSB/pytrendy/commit/35beed45932208dae6a749434227a2ad5e8a514f))

## [1.0.2](https://github.com/RussellSB/pytrendy/compare/v1.0.1...v1.0.2) (2025-08-31)


### Bug Fixes

* **ci:** setting environment to release ([9a8a983](https://github.com/RussellSB/pytrendy/commit/9a8a98330d7eaf19e137046fdbc8ece3f3def7d7))

## [1.0.1](https://github.com/RussellSB/pytrendy/compare/v1.0.0...v1.0.1) (2025-08-31)


### Bug Fixes

* **ci:** add poetry for npm build semantic release ([9049260](https://github.com/RussellSB/pytrendy/commit/9049260781838917ec46b8cb2b6a0fb46e7c8ffe))
* **ci:** install conventional changelog dependancy ([49a1a64](https://github.com/RussellSB/pytrendy/commit/49a1a649813404631e165d4e2823ee9d12d280de))
* **ci:** output from semantic release commit scan and enable version bumping ([c863285](https://github.com/RussellSB/pytrendy/commit/c86328560ff8491cd207000e6477d929e8f83d7b))

# 1.0.0 (2025-08-31)


### Bug Fixes

* **ci:** updating with .releaserc for version bumping ([625b8ee](https://github.com/RussellSB/pytrendy/commit/625b8eecd63931df9c9762f9a015a97a118c620a))
* **clean_artifacts:** Catering for 1 day abrupt period, by relaxing to < 1 filtering ([5ede69f](https://github.com/RussellSB/pytrendy/commit/5ede69f8787e8cee3403a1aa864a0036e5e995eb))
* **dummy-commit:** testing new ci version bumping ([d23546e](https://github.com/RussellSB/pytrendy/commit/d23546e68cbec8af2ebbdf5308af272ad52925bf))


### Features

* Toggle abrupt padding functionality. Allows you to pad abrupt trends for quasi-experimental pre vs post use cases ([8832469](https://github.com/RussellSB/pytrendy/commit/8832469e81e359252c7ad103cc08967082ba281e))
