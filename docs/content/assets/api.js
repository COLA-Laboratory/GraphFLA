window.MathJax = {tex: {inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]]}};
function revealTarget() {
  if (!location.hash) return;
  const id = decodeURIComponent(location.hash.slice(1));
  const target = document.getElementById(id);
  if (!target) return;
  let parent = target.parentElement;
  while (parent) {
    if (parent.tagName === 'DETAILS') {
      parent.open = true;
      if (parent.hidden && parent.classList.contains('method-detail')) {
        const panel = parent.closest('.tabbed-block');
        panel.querySelector('.method-filter').value = '';
        panel.querySelectorAll('.method-detail').forEach(item => { item.hidden = false; });
      }
    }
    if (parent.classList.contains('tabbed-block')) {
      const container = parent.closest('.tabbed-set');
      const blocks = [...parent.parentElement.children];
      const inputs = [...container.children].filter(e => e.tagName === 'INPUT');
      const input = inputs[blocks.indexOf(parent)];
      if (input && !input.checked) {
        input.checked = true;
        input.dispatchEvent(new Event('change', {bubbles: true}));
      }
    }
    parent = parent.parentElement;
  }
  const destination = target.classList.contains('member-anchor')
    ? (target.closest('details') || target) : target;
  requestAnimationFrame(() => destination.scrollIntoView({block: 'start', behavior: 'instant'}));
}
window.addEventListener('hashchange', revealTarget);
// Equations and web fonts can move an authored section after the initial jump.
const initialHash = location.hash;
window.addEventListener('load', () => {
  Promise.all([document.fonts?.ready, window.MathJax?.startup?.promise]).then(() => {
    if (location.hash === initialHash) revealTarget();
  });
});
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.md-nav--primary .md-nav__link').forEach(link => {
    const label = link.querySelector('.md-ellipsis');
    if (label) link.title = label.textContent.trim();
  });
  revealTarget();
  document.querySelectorAll('.method-filter').forEach(input => {
    input.addEventListener('input', () => {
      const query = input.value.toLowerCase().trim();
      input.closest('.tabbed-block').querySelectorAll('.method-detail').forEach(item => {
        item.hidden = !item.querySelector('summary').textContent.toLowerCase().includes(query);
      });
    });
  });
  document.querySelectorAll('.expand-methods').forEach(button => {
    button.addEventListener('click', () => {
      const items = [...button.closest('.tabbed-block').querySelectorAll('.method-detail:not([hidden])')];
      const expand = items.some(item => !item.open);
      items.forEach(item => item.open = expand);
      button.textContent = expand ? 'Collapse all' : 'Expand all';
    });
  });
});
