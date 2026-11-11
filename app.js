(() => {
  const $ = (id) => document.getElementById(id);

  const pct = (x) => {
    if (x == null || Number.isNaN(x)) return "n/a";
    return `${(x * 100).toFixed(1).replace(/\.0$/, "")}%`;
  };

  const usd = (n) =>
    Number(n).toLocaleString("en-US", { style: "currency", currency: "USD" });

  const daysWord = (n) => (n === 1 ? "1 day" : `${n} days`);

  function drawChart(svg, series, day) {
    const W = 820;
    const H = 390;
    const pad = { l: 56, r: 24, t: 18, b: 46 };
    const iw = W - pad.l - pad.r;
    const ih = H - pad.t - pad.b;
    // Leave a little room above 100% so a flat line at 100% is not drawn on the frame.
    const yMax = 1.06;
    const x = (d) => pad.l + (d / 56) * iw;
    const y = (v) => pad.t + ((yMax - v) / yMax) * ih;

    const path = (key) =>
      series
        .map((s, i) => `${i === 0 ? "M" : "L"}${x(s.day).toFixed(1)},${y(s[key]).toFixed(1)}`)
        .join(" ");

    const now0 = series[0].now_apparent_recall;
    const wait0 = series[0].wait_apparent_recall;
    const gapPath = [
      `M${x(0)},${y(now0)}`,
      ...series.map((s) => `L${x(s.day)},${y(s.now_apparent_recall)}`),
      ...[...series].reverse().map((s) => `L${x(s.day)},${y(s.wait_apparent_recall)}`),
      "Z",
    ].join(" ");

    const yTicks = [0, 0.25, 0.5, 0.75, 1];
    const xTicks = [0, 14, 21, 28, 42, 56];
    const cur = series[day];

    const grid = yTicks
      .map(
        (t) =>
          `<line class="chart-grid" x1="${pad.l}" x2="${W - pad.r}" y1="${y(t)}" y2="${y(t)}" />
           <text class="chart-label" x="${pad.l - 8}" y="${y(t) + 4}" text-anchor="end">${pct(t)}</text>`
      )
      .join("");

    const xgrid = xTicks
      .map(
        (t) =>
          `<line class="chart-tick" x1="${x(t)}" x2="${x(t)}" y1="${pad.t + ih}" y2="${pad.t + ih + 5}" />
           <text class="chart-label" x="${x(t)}" y="${H - 16}" text-anchor="middle">${t}d</text>`
      )
      .join("");

    svg.innerHTML = `
      <title>Apparent recall as disputes arrive</title>
      ${grid}
      ${xgrid}
      <line class="chart-axis" x1="${pad.l}" x2="${pad.l}" y1="${pad.t}" y2="${pad.t + ih}" />
      <line class="chart-axis" x1="${pad.l}" x2="${W - pad.r}" y1="${pad.t + ih}" y2="${pad.t + ih}" />
      <text class="chart-label" x="${pad.l + iw / 2}" y="${H - 2}" text-anchor="middle">Days after the evaluation week</text>
      <text class="chart-label" transform="translate(14 ${pad.t + ih / 2}) rotate(-90)" text-anchor="middle">Apparent recall</text>
      <path class="gap-fill" d="${gapPath}"></path>
      <path class="line-true" d="M${x(0)},${y(series[0].now_true_recall)} L${x(56)},${y(series[0].now_true_recall)}"></path>
      <path class="line-true" d="M${x(0)},${y(series[0].wait_true_recall)} L${x(56)},${y(series[0].wait_true_recall)}"></path>
      <path class="line-now" d="${path("now_apparent_recall")}"></path>
      <path class="line-wait" d="${path("wait_apparent_recall")}"></path>
      <line class="cursor-line" x1="${x(day)}" x2="${x(day)}" y1="${pad.t}" y2="${pad.t + ih}"></line>
      <circle class="dot-now" cx="${x(day)}" cy="${y(cur.now_apparent_recall)}" r="4.5"></circle>
      <circle class="dot-wait" cx="${x(day)}" cy="${y(cur.wait_apparent_recall)}" r="4.5"></circle>
    `;
  }

  function metricsHtml(cur, summary) {
    return `
      <div class="metric">
        <div class="lbl">Reports in</div>
        <div class="val">${cur.fraud_visible}<span style="font-size:1rem;color:var(--ink-soft)"> / ${cur.fraud_total}</span></div>
        <div class="sub">Fraud cases already reported</div>
      </div>
      <div class="metric">
        <div class="lbl">Fit now, apparent</div>
        <div class="val now">${pct(cur.now_apparent_recall)}</div>
        <div class="sub">True recall ${pct(cur.now_true_recall)}</div>
      </div>
      <div class="metric">
        <div class="lbl">Wait 6 weeks, apparent</div>
                <div class="val wait">${pct(cur.wait_apparent_recall)}</div>
        <div class="sub">True recall ${pct(cur.wait_true_recall)}. ${summary.legit_flagged_wait} false alarms vs ${summary.legit_flagged_now} for Fit now</div>
      </div>
      <div class="metric">
        <div class="lbl">Label-lag gap</div>
        <div class="val now">${pct(summary.now_label_lag_gap)}</div>
                <div class="sub">Fit now: day 0 minus day 56. Wait: ${pct(summary.wait_label_lag_gap)}</div>
      
      </div>`;
  }

  function reported(p, asOf) {
    if (!p.is_fraud) return false;
    return p.day + p.report_delay <= asOf;
  }

  function barRows(parts) {
    const max = Math.max(0.01, ...parts.map((p) => Math.abs(p.contribution)));
    return parts
      .slice(0, 6)
      .map((p) => {
        const raise = p.contribution >= 0;
        const w = (Math.abs(p.contribution) / max) * 50;
        const left = raise ? 50 : 50 - w;
        const cls = raise ? "raise" : "lower";
        const sign = raise ? "+" : "";
        return `<div class="bar-row">
          <div class="name">${p.label}</div>
          <div class="track"><span class="${cls}" style="left:${left}%;width:${w}%"></span></div>
          <div class="n">${sign}${p.contribution.toFixed(2)}</div>
        </div>`;
      })
      .join("");
  }

  function renderCase(data, payment, day) {
    const asOf = data.summary.eval_end_day + day;
    const arrived = reported(payment, asOf);
    const delay = payment.report_delay;
    const related = (data.related || []).filter((r) => r.related_to === payment.id);
    const explain = payment.explain_now || payment.explain_wait;
    const parts = explain ? explain.parts : [];

    let verdict;
    if (payment.kind === "obvious") {
      verdict = `A fast, obvious pattern: a throwaway email, a new card, and a small test-sized charge. Both rules flagged it. The report arrived after ${daysWord(delay)}.`;
        } else if (!payment.flagged_now && payment.flagged_wait) {
      const hid = (e) => ((e && e.parts) || []).find((x) => x.id === "history_but_new_here");
      const hn = hid(payment.explain_now), hw = hid(payment.explain_wait);
            const CALM = {
        velocity: "only one or two charges this hour", amount_shape: "a normal amount", network_history: "years of clean history",
        ip_cards: "few cards on this IP", merchant_dispute: "a low-dispute merchant", new_email: "an established email",
        new_card: "an established card", throwaway_email: "a normal email address", country_mismatch: "matching countries",
        name_mismatch: "a name that matches the email",
      };
      const downs = parts.filter((x) => x.contribution < 0 && CALM[x.id]).slice(0, 2).map((x) => CALM[x.id]);
      const why = hn && hw
        ? ` Both models saw the hiding pattern, clean history elsewhere and a first charge here. Fit now gave it less weight (+${hn.contribution.toFixed(1)} vs +${hw.contribution.toFixed(1)}), because many recent charges like this were still labeled legitimate when it trained.${downs.length ? ` Ordinary details (${downs.join(" and ")}) were enough to keep it under the line.` : ""}`
        : "";
      verdict = `Fit now left this charge through. The model trained only on mature weeks flagged it. The cardholder reported the dispute ${daysWord(delay)} later.${why}`;
    } else if (payment.flagged_now && payment.flagged_wait) {
      verdict = `Both rules flagged this stolen-card charge. The dispute itself did not arrive for ${daysWord(delay)}.`;
    } else {
      verdict = `Neither rule flagged this charge. The dispute arrived ${daysWord(delay)} later.`;
    }

    const flags = [
      payment.flagged_now
        ? `<span class="flag hit">Fit now flagged it</span>`
        : `<span class="flag miss">Fit now left it through</span>`,
      payment.flagged_wait
        ? `<span class="flag hit">Wait 6 weeks flagged it</span>`
        : `<span class="flag miss">Wait 6 weeks left it through</span>`,
      arrived
        ? `<span class="flag">Dispute has arrived by day ${day}</span>`
        : `<span class="flag">Dispute has not arrived yet</span>`,
    ].join("");

    const relTable = related.length
      ? `<div class="related">
          <h3>Earlier charges on the same card at other merchants</h3>
          <table>
            <thead><tr><th>Day</th><th>Merchant</th><th>Amount</th><th>Shared</th></tr></thead>
            <tbody>
              ${related
                .map(
                  (r) => `<tr>
                    <td>${r.day}</td>
                    <td>${r.merchant}</td>
                    <td>${usd(r.amount_usd)}</td>
                    <td>${r.share}</td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>
        </div>`
      : "";

    return `
      <p class="verdict">${verdict}</p>
      <div class="flags">${flags}</div>
      <dl class="facts">
        <div><dt>Merchant</dt><dd>${payment.merchant}</dd></div>
        <div><dt>Amount</dt><dd>${usd(payment.amount_usd)}</dd></div>
        <div><dt>Card</dt><dd>${payment.card}</dd></div>
        <div><dt>Email</dt><dd>${payment.email}</dd></div>
        <div><dt>IP</dt><dd>${payment.ip} (${payment.city}, ${payment.ip_country})</dd></div>
        <div><dt>Billing country</dt><dd>${payment.billing_country}</dd></div>
        <div><dt>Clean history elsewhere</dt><dd>${payment.network_good_years} years</dd></div>
        <div><dt>First time at this merchant</dt><dd>${payment.first_time_at_merchant ? "Yes" : "No"}</dd></div>
        <div><dt>Scored on sample day</dt><dd>${payment.day}</dd></div>
        <div><dt>Dispute delay</dt><dd>${daysWord(delay)}</dd></div>
        <div><dt>Fit now score</dt><dd>${payment.score_now.toFixed(3)}</dd></div>
        <div><dt>Wait 6 weeks score</dt><dd>${payment.score_wait.toFixed(3)}</dd></div>
      </dl>
      <div class="explain">
                <h3>What pushed the Fit now score up or down</h3>
        <div class="bars">${barRows(parts)}</div>
        <p class="bars-note">Pushes are on the log-odds scale, measured against an average charge. The model's starting point is left out. It is strongly negative because most charges are legitimate, so a charge is flagged only when the pushes add up past the line.</p>
      </div>
      ${relTable}`;
  }

  function renderCases(data, selectedId, day, onPick) {
    const asOf = data.summary.eval_end_day + day;
    const late = data.payments
      .filter((p) => p.is_fraud && p.kind === "camouflage")
      .sort((a, b) => a.report_delay - b.report_delay || a.day - b.day);

    $("cases").innerHTML = late
      .map((p) => {
        const status = !p.flagged_now && p.flagged_wait
          ? "Missed by fit now"
          : p.flagged_now
            ? "Caught by both"
            : "Missed by both";
        const arrived = reported(p, asOf) ? "reported" : "still unlabeled";
        return `<button type="button" role="option" data-id="${p.id}" class="${p.id === selectedId ? "is-on" : ""}" aria-selected="${p.id === selectedId}">
          <span class="who">${p.name} &middot; ${usd(p.amount_usd)}</span>
          <span class="pill">${status}</span>
          <span class="meta">${p.merchant} &middot; dispute in ${daysWord(p.report_delay)} &middot; ${arrived}</span>
        </button>`;
      })
      .join("");

    $("cases").querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => onPick(btn.dataset.id));
    });
  }

  function mount(data) {
    $("disclaimer").textContent = data.disclaimer;
    const slider = $("day");
    const series = data.series;
    const late = data.payments.filter((p) => p.is_fraud && p.kind === "camouflage");
    const firstMiss = late.find((p) => !p.flagged_now && p.flagged_wait) || late[0];
    let selected = firstMiss ? firstMiss.id : late[0].id;
    let day = 28;

    slider.value = String(day);

    const paint = () => {
      day = Number(slider.value);
      const cur = series[day];
      $("day-label").textContent = String(day);
      $("readout").textContent =
        `Day ${day}: ${cur.fraud_visible} of ${cur.fraud_total} reports in · Fit now ${pct(cur.now_apparent_recall)} · Wait ${pct(cur.wait_apparent_recall)}`;
      drawChart($("chart"), series, day);
      $("metrics").innerHTML = metricsHtml(cur, data.summary);
      const payment = data.payments.find((p) => p.id === selected);
      renderCases(data, selected, day, (id) => {
        selected = id;
        paint();
      });
      $("case").innerHTML = payment
        ? renderCase(data, payment, day)
        : `<p class="empty">Select a charge to inspect it.</p>`;
    };

    slider.addEventListener("input", paint);
    paint();
  }

  fetch("./data/sample.json")
    .then((r) => {
      if (!r.ok) throw new Error("Could not load the sample.");
      return r.json();
    })
    .then(mount)
    .catch((err) => {
      $("readout").textContent = err.message;
    });
})();
