# Paper draft (ACM sigconf)

Build with a TeX Live that has `acmart` (or Overleaf's ACM template):

    cd paper && latexmk -pdf main.tex

Every number in the paper comes from `benchmarks/results/*.json`. The CIFAR-10 review of
flagged pairs that are not on the ciFAIR list is in `benchmarks/results/cifar_unlisted_review.csv`
(contact sheet: `cifar_unlisted_T4.png`); it was done by one reviewer and should be checked by a
second before submission.
