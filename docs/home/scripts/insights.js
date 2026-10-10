/* Recorded research results; D3 owns scales, axes and transitions. */
(() => {
  const data = window.graphflaInsights || [];
  if (!window.d3 || !document.querySelector('[data-insight]')) return;
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const valueText = d3.format('.3~g');
  const duration = matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 240;

  function domain(values) {
    const [low, high] = d3.extent(values);
    const pad = low === high ? Math.max(Math.abs(low) * 0.1, 0.1) : (high - low) * 0.08;
    return [low - pad, high + pad];
  }
  function regression(points) {
    if (points.length < 3) return null;
    const mx = d3.mean(points, d => d.x), my = d3.mean(points, d => d.y);
    const xx = d3.sum(points, d => (d.x - mx) ** 2);
    if (!xx) return null;
    const slope = d3.sum(points, d => (d.x - mx) * (d.y - my)) / xx;
    return x => my + slope * (x - mx);
  }
  function wrap(selection, label, width) {
    selection.text(null);
    let line = [], tspan = selection.append('tspan').attr('x', selection.attr('x')).attr('dy', 0);
    for (const word of label.split(' ')) {
      line.push(word); tspan.text(line.join(' '));
      if (line.length > 1 && tspan.node().getComputedTextLength() > width) {
        line.pop(); tspan.text(line.join(' ')); line = [word];
        tspan = selection.append('tspan').attr('x', selection.attr('x')).attr('dy', '1.15em').text(word);
      }
    }
  }

  document.querySelectorAll('[data-insight]').forEach(card => {
    const source = data.find(d => d.id === card.dataset.insight);
    if (!source) return;
    const feature = card.querySelector('[data-feature]');
    const outcome = card.querySelector('[data-outcome]');
    const canvas = card.querySelector('.gfl-insight__canvas');
    const svg = d3.select(card.querySelector('svg'));
    const detail = card.querySelector('.gfl-insight__detail');
    const publication = detail.querySelector('[data-publication]');
    let pinned = false, hideTimer, currentPoint, stopPositioning, points = [], currentWidth = 0;

    function close() {
      clearTimeout(hideTimer); pinned = false; detail.hidden = true;
      if (stopPositioning) { stopPositioning(); stopPositioning = null; }
      svg.selectAll('circle').attr('aria-expanded', 'false');
    }
    function delayClose() {
      if (!pinned) hideTimer = setTimeout(close, 250);
    }
    function show(node, point, pin = false) {
      clearTimeout(hideTimer); currentPoint = node; pinned = pin;
      const paper = point.record.publication || {};
      const xName = feature.selectedOptions[0].textContent;
      const yName = outcome.selectedOptions[0].textContent;
      detail.querySelector('[data-dataset]').textContent = point.record.label;
      const yText = source.id === 'evolution' ? d3.format('.3%')(point.y) : valueText(point.y);
      detail.querySelector('[data-values]').textContent = `${xName}: ${valueText(point.x)} · ${yName}: ${yText}`;
      detail.querySelector('[data-paper]').textContent = paper.title || point.record.id;
      detail.querySelector('[data-journal]').textContent = [paper.journal, paper.year].filter(Boolean).join(' · ');
      const url = paper.url || (paper.doi ? `https://doi.org/${paper.doi}` : point.record.source_url);
      publication.hidden = !url;
      if (url && /^https?:\/\//.test(url)) publication.href = url;
      else publication.removeAttribute('href');
      detail.hidden = false;
      svg.selectAll('circle').attr('aria-expanded', 'false');
      d3.select(node).attr('aria-expanded', 'true');
      if (stopPositioning) stopPositioning();
      const {computePosition, offset, flip, shift, autoUpdate} = FloatingUIDOM;
      const mark = node.getBoundingClientRect(), box = canvas.getBoundingClientRect();
      const placement = mark.left < box.left + box.width / 2 ? 'right' : 'left';
      const position = () => computePosition(node, detail, {
        placement,
        middleware: [offset(14), flip({boundary: card, padding: 12,
          fallbackPlacements: [placement === 'right' ? 'left' : 'right', 'top', 'bottom']}),
          shift({padding: 12})],
      }).then(({x, y}) => {
        if (currentPoint !== node || detail.hidden) return;
        detail.style.setProperty('--tip-x', `${x}px`);
        detail.style.setProperty('--tip-y', `${y}px`);
      });
      stopPositioning = autoUpdate(node, detail, position);
    }
    detail.addEventListener('pointerenter', () => clearTimeout(hideTimer));
    detail.addEventListener('pointerleave', delayClose);
    detail.addEventListener('focusin', () => clearTimeout(hideTimer));
    detail.addEventListener('focusout', event => { if (!detail.contains(event.relatedTarget)) delayClose(); });
    function dismiss() {
      close(); if (currentPoint?.isConnected) currentPoint.focus({preventScroll: true}); close();
    }
    detail.querySelector('[data-close]').addEventListener('click', dismiss);
    card.addEventListener('keydown', event => { if (event.key === 'Escape' && !detail.hidden) dismiss(); });
    document.addEventListener('pointerdown', event => {
      const point = event.target.closest?.('.gfl-insight__points circle');
      if (!detail.contains(event.target) && (!point || !card.contains(point))) close();
    });

    function render(animate = true) {
      close();
      const width = Math.max(220, canvas.clientWidth), height = Math.min(350, Math.max(300, width * 0.55));
      currentWidth = width;
      const margin = {left: source.id === 'evolution' ? 92 : 60, right: 16, top: 20, bottom: width < 380 ? 70 : 58};
      const right = width - margin.right, bottom = height - margin.bottom;
      const fw = right - margin.left, fh = bottom - margin.top;
      points = source.records.map(record => ({record, x: record.features[feature.value], y: record.outcomes[outcome.value]}))
        .filter(d => finite(d.x) && finite(d.y));
      svg.attr('viewBox', `0 0 ${width} ${height}`);
      svg.select('clipPath rect').attr('x', margin.left).attr('y', margin.top).attr('width', fw).attr('height', fh);
      const empty = svg.select('.gfl-insight__empty').attr('x', margin.left + fw / 2).attr('y', margin.top + fh / 2);
      card.querySelector('[data-count]').textContent = `${points.length === source.records.length ? points.length : `${points.length} of ${source.records.length}`} DMS datasets`;
      const label = feature.selectedOptions[0].textContent;
      const xLabel = svg.select('[data-label=x]').attr('x', margin.left + fw / 2).attr('y', height - (width < 380 ? 32 : 14));
      wrap(xLabel, label, Math.max(180, fw));
      const cy = margin.top + fh / 2;
      svg.select('[data-label=y]').attr('x', 16).attr('y', cy).attr('transform', `rotate(-90 16 ${cy})`);
      if (!points.length) {
        svg.selectAll('.gfl-insight__axis, .gfl-insight__grid, .gfl-insight__points').selectAll('*').remove();
        svg.select('.gfl-insight__fit').attr('d', null);
        card.querySelector('.gfl-insight__footer > :last-child').hidden = true;
        empty.text('No paired results for this selection');
        return;
      }
      empty.text('');
      const x = d3.scaleLinear().domain(domain(points.map(d => d.x))).nice().range([margin.left, right]);
      const y = d3.scaleLinear().domain(domain(points.map(d => d.y))).nice().range([bottom, margin.top]);
      const yTicks = y.ticks(5).filter(tick => tick >= (source.id === 'proteingym' ? -1 : 0) && tick <= 1);
      const ticks = width < 380 ? 3 : 5;
      const transition = svg.transition().duration(animate ? duration : 0).ease(d3.easeCubicOut);
      svg.select('[data-axis=x]').attr('transform', `translate(0,${bottom})`).transition(transition)
        .call(d3.axisBottom(x).ticks(ticks, '~g').tickSizeOuter(0));
      const yAxis = d3.axisLeft(y).ticks(5, source.y_format || '~g').tickValues(yTicks).tickSizeOuter(0);
      svg.select('[data-axis=y]').attr('transform', `translate(${margin.left},0)`).transition(transition).call(yAxis);
      svg.select('.gfl-insight__grid').attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).tickValues(yTicks).tickSize(-fw).tickFormat(''));
      const fit = regression(points), extent = d3.extent(points, d => d.x);
      const line = fit ? d3.line()([[x(extent[0]), y(fit(extent[0]))], [x(extent[1]), y(fit(extent[1]))]]) : null;
      svg.select('.gfl-insight__fit').transition(transition).attr('d', line);
      card.querySelector('.gfl-insight__footer > :last-child').hidden = !fit;
      svg.select('.gfl-insight__points').selectAll('circle').data(points, d => d.record.id)
        .join(enter => enter.append('circle').attr('r', 0), update => update, exit => exit.remove())
        .attr('role', 'button').attr('tabindex', (_, i) => i === 0 ? 0 : -1)
        .attr('aria-expanded', 'false')
        .attr('aria-label', d => `${d.record.label}. ${label}: ${valueText(d.x)}. ${outcome.selectedOptions[0].textContent}: ${source.id === 'evolution' ? d3.format('.3%')(d.y) : valueText(d.y)}. Publication details.`)
        .on('pointerenter', function(event, d) { if (!pinned) show(this, d); })
        .on('pointerleave', delayClose)
        .on('focus', function(event, d) { show(this, d); })
        .on('blur', delayClose)
        .on('click', function(event, d) { show(this, d, true); })
        .on('keydown', function(event, d) {
          const nodes = svg.selectAll('.gfl-insight__points circle').nodes(), index = nodes.indexOf(this);
          const step = ['ArrowRight', 'ArrowDown'].includes(event.key) ? 1 : ['ArrowLeft', 'ArrowUp'].includes(event.key) ? -1 : 0;
          if (step) {
            event.preventDefault(); const target = nodes[(index + step + nodes.length) % nodes.length];
            nodes.forEach(n => n.tabIndex = n === target ? 0 : -1); target.focus();
          } else if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault(); show(this, d, true); publication.focus({preventScroll: true});
          }
        })
        .transition(transition).attr('cx', d => x(d.x)).attr('cy', d => y(d.y)).attr('r', width < 380 ? 3.5 : 4);
    }
    feature.addEventListener('change', () => render());
    outcome.addEventListener('change', () => render());
    [feature, outcome].forEach(select => {
      if (!window.TomSelect) return;
      const control = new TomSelect(select, {
        plugins: ['dropdown_input'], create: false, maxOptions: null, refreshThrottle: 0,
        closeAfterSelect: true, onDelete: () => false,
        onDropdownOpen: close,
      });
      control.control.setAttribute('aria-label', select.getAttribute('aria-label'));
      select.parentElement.querySelector('.dropdown-input').setAttribute('aria-label', `Search ${select.getAttribute('aria-label').toLowerCase()}`);
      select.parentElement.querySelector('.dropdown-input').placeholder = 'Search…';
    });
    new ResizeObserver(() => { if (Math.abs(canvas.clientWidth - currentWidth) > 1) render(false); }).observe(canvas);
    render(false);
  });
})();
