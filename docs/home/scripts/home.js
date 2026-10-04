// Preserve bookmarks to the Landscape API that used to live at the site root.
if (document.querySelector(".gfl-home") && location.hash.startsWith("#graphfla.landscape.")) {
  location.replace(new URL("landscape/" + location.hash, location.href).href);
}

// Copy buttons: any element with data-copy puts that text on the clipboard.
document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-copy]");
  if (button && navigator.clipboard) {
    const label = button.getAttribute("aria-label");
    try {
      await navigator.clipboard.writeText(button.dataset.copy);
      button.setAttribute("aria-label", "Copied");
      button.title = "Copied";
    } catch {
      button.setAttribute("aria-label", "Select and copy the command");
      button.title = "Select and copy the command";
    }
    window.setTimeout(() => {
      button.setAttribute("aria-label", label);
      button.removeAttribute("title");
    }, 2000);
  }
});

// A tab changes its input table, graph and computed features as one unit.
document.querySelectorAll('.gfl-scenario-tabs').forEach((group) => {
  const tabs = [...group.querySelectorAll('[role="tab"]')];
  const activate = (tab) => {
    tabs.forEach((item) => {
      const selected = item === tab;
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
      document.getElementById(item.getAttribute('aria-controls')).hidden = !selected;
    });
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', (event) => {
      const target = event.key === 'ArrowRight' ? (index + 1) % tabs.length
        : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : null;
      if (target === null) return;
      event.preventDefault();
      activate(tabs[target]);
      tabs[target].focus();
    });
  });
});
