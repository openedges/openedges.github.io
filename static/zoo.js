/* Search across every task table. No dependencies.
 *
 * The rows are already rendered server-side, so this only hides them: the page is fully
 * readable with JavaScript off, which also means it works from a file:// URL where fetching
 * a sibling .json would be blocked. */

(function () {
  "use strict";

  var q = document.getElementById("q");
  var reset = document.getElementById("reset");
  var tally = document.getElementById("tally");
  if (!q) return;

  var tables = [].slice.call(document.querySelectorAll(".task")).map(function (section) {
    return {
      section: section,
      shown: section.querySelector(".shown"),
      empty: section.querySelector(".empty"),
      rows: [].slice.call(section.querySelectorAll("tbody tr")).map(function (tr) {
        return { tr: tr, search: tr.getAttribute("data-search") || "" };
      })
    };
  });

  function apply() {
    var needle = q.value.trim().toLowerCase();
    var words = needle ? needle.split(/\s+/) : [];
    var total = 0;
    var visibleSections = 0;

    tables.forEach(function (table) {
      var matched = 0;
      table.rows.forEach(function (row) {
        var hit = words.every(function (word) { return row.search.indexOf(word) !== -1; });
        row.tr.hidden = !hit;
        if (hit) matched += 1;
      });

      if (table.shown) table.shown.textContent = String(matched);
      if (table.empty) table.empty.hidden = matched !== 0;
      table.section.hidden = matched === 0 && words.length > 0;

      total += matched;
      if (!table.section.hidden) visibleSections += 1;
    });

    tally.textContent = words.length
      ? total + " of " + countAll() + " models match, in " + visibleSections + " task(s)"
      : "";
  }

  function countAll() {
    return tables.reduce(function (sum, table) { return sum + table.rows.length; }, 0);
  }

  q.addEventListener("input", apply);
  if (reset) {
    reset.addEventListener("click", function () {
      q.value = "";
      apply();
    });
  }

  apply();
})();
