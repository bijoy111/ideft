# Online backup selection (held for a follow-up paper)

These files were Section VII and the appendix of the backup-selection paper:
online re-selection under churn, the competitive analysis, and the measured
migration cost. They were removed from `paper.tex` to keep that paper to
placement and measurement, and are kept here unchanged.

They are not built by `paper.tex`. `make_online_tables.py` and
`make_migration_tables.py` still write `tables_online.tex` and
`tables_migration.tex` to the parent directory; move the output here.

Open points from review, to settle before this becomes a paper:

- Table `tab:online-churn` uses the assumed M = 3, not the measured 23-41.
- The text calls SNAPSHOT worst at every setting; STATIC and PERIODIC are worse
  from p_deg = 0.02 upward.
- Results for alpha = 4 and 8 are cited but the table shows alpha = 1 and 2.
- The reported ratios (5.9 at epsilon = 0.1) do not match the proof's 2M/epsilon.
- For uniform metrical task systems the randomized bounds are Theta(log k);
  check the Omega(log k / log log k) and O(log^2 k) citations, which look like
  the general-metric results.
- Theorem 1 is a standard result and should be presented as such.
