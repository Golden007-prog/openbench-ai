# Glossary

## Confidence interval

A confidence interval shows the uncertainty around an estimate, such as the average score difference between two models. Over many equivalent studies, a well-calibrated 95% interval procedure would cover the true value about 95% of the time. OpenBench's paired bootstrap resamples whole items so the two scores for an item stay together.

See [Reading the difference and confidence interval](reading_a_report.md#3-read-the-difference-and-confidence-interval).

## Descriptive only

A descriptive-only result summarizes the observed scores without a formal test decision for that result. For example, a category breakdown can give a mean difference and a bootstrap interval while leaving out its adjusted p-value. An exploratory report can also be descriptive-only when the inputs fail compatibility checks.

See [What `descriptive_only` means on breakdowns](model_comparison.md#what-descriptive_only-means-on-breakdowns).

## Discordance

For paired binary scores, an item is discordant when one model succeeds and the other fails. The discordance rate is the share of items where that happens in either direction. Higher discordance generally means more items are needed to detect the same accuracy difference.

See [Sample-size planning](ranking_and_planning.md#openbench-plan).

## Holm correction

Holm correction adjusts a group of p-values to control the chance of at least one false positive across the group. It works through the ordered p-values, starting with the smallest. OpenBench adjusts difference tests and equivalence tests in separate families.

See [Multiple comparisons](statistical_methodology.md#6-multiple-comparisons).

## Paired comparison

A paired comparison matches the two models' scores for each shared dataset item and analyzes the within-item differences. When an arm is a repeat group, OpenBench first averages its valid scores for that item. Each aligned item still contributes one difference.

See [Alignment and missing data](model_comparison.md#alignment-and-missing-data).

## Rank interval

A rank interval gives the best and worst ranks supported by the pairwise comparisons. After Holm correction, the best rank is one plus the number of models that scored significantly better, and the worst rank is the total number of models minus the number that scored significantly worse. A wide interval means the data leave several rankings plausible.

See [Ranking several models](statistical_methodology.md#18-ranking-several-models-openbench-rank-v050).

## Repeat group

A repeat group collects two or more runs from a repeated evaluation of one model and records their shared configuration and lineage. Repetitions show how much scores vary across runs. In a paired analysis, OpenBench averages them per item so they do not inflate the sample size.

See [Repeated runs](repeated_runs.md).

## TOST (two one-sided tests)

TOST checks whether a difference fits inside an equivalence margin chosen before the analysis. One test checks the lower bound and the other checks the upper bound, and both must pass with the applicable multiplicity adjustment. A non-significant difference test by itself leaves equivalence unresolved.

See [Significance, practical importance and equivalence](statistical_methodology.md#7-significance-practical-importance-and-equivalence-are-separate).
