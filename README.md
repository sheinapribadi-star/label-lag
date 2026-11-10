# Label Lag

A fraud model can look better than it is, because the truth is still in the mail.

**Live demo:** https://sheinapribadi-star.github.io/label-lag/

This is a small, interactive study of delayed fraud labels. You score a week of payments, then drag a slider through the following eight weeks as disputes arrive. Apparent recall starts high, when only the obvious cases have been reported. It then moves toward the true number as stolen-card reports come in.

## Source

The project starts from [Scaling Radar: Fighting fraud with agent-assisted machine learning](https://stripe.dev/blog/scaling-radar-fighting-fraud-with-agent-assisted-machine-learning) by Ryan Drapeau and Sai Samant (Stripe, 19 October 2026).

In that post, Stripe trained a Radar model on a much larger slice of its network. The new model looked better in offline tests and in the early days of a production experiment. Then more fraud reports arrived, measured performance fell below the earlier model, and they rolled it back.

The reason is simple and easy to miss. When a stolen card is used, it can take weeks for the cardholder to notice and report it to their bank. Until Stripe receives that dispute, the payment sits in the training and evaluation data as a legitimate charge. Recent weeks, which you want because they show new attacks, are also the weeks whose labels you can trust the least.

A related 2023 post, [How we built it: Stripe Radar](https://stripe.dev/blog/how-we-built-it-stripe-radar), makes a second point this demo uses: explanation matters as much as detection. Stripe built a view of the features that raised or lowered a payment's fraud score, so a reviewer can see why a charge looked risky, or why it looked safe.

This project is not a reimplementation of Radar, and it does not use Stripe data.

## The idea, in plain English

Imagine you are reviewing a model on Monday. Of the fraud you know about so far, it caught almost all of it. Two weeks later the same model, scored the same way, looks worse. Nothing about the model changed. The labels did. Stolen-card disputes arrived for charges the model had treated as legitimate, both when it was trained and when it was first evaluated.

The hard cases in this sample follow a pattern from the Stripe post: a card that is new to one business can still have years of clean history elsewhere on the network. That history pushes the score down. The charge looks ordinary. The report arrives a month later.

## Design decisions

Two training rules, both taken from the tradeoff in the post.

1. **Fit now, with the labels you have.** Train on every payment in a 10-week window, using only the reports that have arrived by training day. Recent weeks count more, so the model can chase new attacks. Those same weeks are where labels are least trustworthy.
2. **Train only on mature weeks.** Drop the most recent six weeks. Train only on payments that are old enough for most disputes to have arrived. Labels are more trustworthy. The window is smaller, and a little stale.

Both rules are logistic models with the same features, including an explicit one for the hiding pattern: clean history on other merchants, combined with a first charge at this merchant. They are scored at a frozen 2% false-positive operating point, chosen on the mature training weeks, where labels can be trusted.

The demo also shows, for each late-reported charge, the features that pushed the "fit now" score up or down, plus earlier charges on the same card at other merchants. That is the explanation table from the 2023 post, applied to the missed-fraud investigation in the 2026 post.

I did not use maps, even though the original Radar write-up mentions billing and shipping maps. City and country are shown as text.

## The metric

The **label-lag gap** is apparent recall on day 0 minus recall once labels have matured (day 56 in this sample), at the frozen 2% false-positive operating point.

A training rule holds up if that gap is small and mature recall stays high. On this constructed sample:

- Fit now starts at 100% apparent recall, then falls to 88.1%. Label-lag gap: 11.9 percentage points. It misses 5 of 28 late-reported stolen-card cases.
- Wait 6 weeks stays at 100% as the labels arrive. Label-lag gap: 0. It catches all 42 evaluation-week fraud cases.

A live system would not go 42 for 42. The useful result is the gap, and the fact that the gap only appears after the disputes arrive. Early evaluation, on its own, would have called the two rules tied.

## About the sample

This is a constructed sample, not Stripe data. The evaluation week has 280 charges, of which 42 are fraud: 14 obvious card-testing cases that get reported quickly, and 28 stolen-card cases that usually take three to eight weeks to be reported. In live card payments, fraud is closer to 1 in 1,000. The sample is deliberately dense so a short demo can show the lag.

All card numbers are masked, and IP addresses use documentation ranges (TEST-NET). Names, merchants, and emails are fictional.

Regenerate the sample:

```bash
python3 generate_data.py
```

Then open the demo with a local server:

```bash
python3 -m http.server 8000
```

Visit http://127.0.0.1:8000/

## Credits

Sheina Pribadi, UC Berkeley junior studying Business and Data Science (Haas + Data Science, '28).

Written after reading Drapeau and Samant, "Scaling Radar: Fighting fraud with agent-assisted machine learning," Stripe, 19 October 2026.
