/* ── Branch indicator (main / develop toggle) ────────────────────────── */
(function () {
    "use strict";

    var GITHUB_PAGES_ROOT = "https://russellsb.github.io/pytrendy/";

    function detectBranch() {
        var h = window.location.hostname;
        if (h === "localhost" || h === "127.0.0.1" || h === "") {
            return "develop";
        }
        var p = window.location.pathname;
        if (/\/main(\/|$)/.test(p))    return "main";
        if (/\/develop(\/|$)/.test(p)) return "develop";
        return "main";
    }

    function isLocal() {
        var h = window.location.hostname;
        return h === "localhost" || h === "127.0.0.1" || h === "";
    }

    function buildToggleUrl(branch) {
        if (isLocal()) {
            /* Local is always "develop"; clicking navigates to the deployed main docs */
            return GITHUB_PAGES_ROOT + "main/";
        }
        var other = branch === "main" ? "develop" : "main";
        /* Swap the branch segment in the path, e.g. /pytrendy/main/ → /pytrendy/develop/ */
        return window.location.href.replace("/" + branch + "/", "/" + other + "/");
    }

    document.addEventListener("DOMContentLoaded", function () {
        var branch = detectBranch();
        var other  = branch === "main" ? "develop" : "main";
        var url    = buildToggleUrl(branch);

        var label = branch === "develop" ? "dev" : "main";
        var otherLabel = other === "develop" ? "dev" : "main";

        var badge = document.createElement("a");
        badge.className = "branch-indicator branch-indicator--" + branch;
        badge.href  = url;
        badge.title = "Switch to " + otherLabel + " docs";
        badge.setAttribute("aria-label",
            "Viewing " + label + " docs — click to switch to " + otherLabel);
        badge.textContent = label;

        var headerNav = document.querySelector(".md-header__inner");
        if (!headerNav) return;

        /* Append after source widget (far right), or append to header */
        var source = headerNav.querySelector(".md-header__source");
        if (source) {
            source.insertAdjacentElement("afterend", badge);
        } else {
            headerNav.appendChild(badge);
        }
    });
}());

/* Auto-expand the "Fundamentals" and "pytrendy" (API Reference) nav sections */
document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".md-nav__link .md-ellipsis").forEach(function (el) {
        var text = el.textContent.trim();
        if (text === "Fundamentals" || text === "pytrendy") {
            var toggle = el.closest(".md-nav__item").querySelector(":scope > .md-nav__toggle");
            if (toggle && !toggle.checked) toggle.checked = true;
        }
    });
});

/* Disable JupyterLite "Leave site?" prompt on navigation */
document.addEventListener("DOMContentLoaded", function () {
    var iframe = document.getElementById("jupyterlite-iframe");
    if (!iframe) return;

    /* ── Wrap iframe in a container and add a loading overlay ── */
    var wrapper = document.createElement("div");
    wrapper.className = "jl-wrapper";
    iframe.parentNode.insertBefore(wrapper, iframe);
    wrapper.appendChild(iframe);

    var overlay = document.createElement("div");
    overlay.className = "jl-loading-overlay";
    overlay.innerHTML = '<div class="spinner"></div> Loading interactive notebook…';
    wrapper.appendChild(overlay);

    function suppressBeforeUnload() {
        try {
            iframe.contentWindow.addEventListener("beforeunload", function (e) {
                e.stopImmediatePropagation();
            }, true);
        } catch (_) {
            /* cross-origin — nothing we can do */
        }
    }

    function hideOverlay() {
        overlay.classList.add("hidden");
        suppressBeforeUnload();
    }

    /* Suppress the "Leave site?" prompt on the parent window too */
    window.addEventListener("beforeunload", function (e) {
        if (document.getElementById("jupyterlite-iframe")) {
            e.stopImmediatePropagation();
            delete e.returnValue;
        }
    }, true);

    iframe.addEventListener("load", function () {
        suppressBeforeUnload();
        setTimeout(hideOverlay, 1000);
    });

    /* Fallback: hide overlay after 10 s even if load event is missed */
    setTimeout(function () { overlay.classList.add("hidden"); }, 10000);
});

/* Forward MkDocs-Material TOC clicks to the JupyterLite iframe via postMessage.
 *
 * The mkdocs-jupyterlite plugin ships its own toc-handler.js, but it targets the
 * selector `#toc-collapse a` which belongs to the old MkDocs default theme.
 * The Material theme stores TOC links inside
 *   [data-md-component="toc"] a[href^="#"]
 * so we attach our own listeners here and the plugin script becomes a no-op
 * (it finds zero matching elements and does nothing).
 */
document.addEventListener("DOMContentLoaded", function () {
    var iframe = document.getElementById("jupyterlite-iframe");
    if (!iframe) return;

    function attachMaterialTocHandlers() {
        /* Only target the secondary nav (right-sidebar TOC), not the mobile
         * copy that lives inside the primary navigation drawer. */
        var tocLinks = document.querySelectorAll(
            'nav.md-nav--secondary [data-md-component="toc"] a[href^="#"]'
        );
        /* Fallback: some Material versions omit the secondary nav wrapper */
        if (!tocLinks.length) {
            tocLinks = document.querySelectorAll('[data-md-component="toc"] a[href^="#"]');
        }
        tocLinks.forEach(function (link) {
            link.addEventListener("click", function (event) {
                event.preventDefault();
                /* Strip excess whitespace that Material's nested <span> adds */
                var headingText = this.textContent.replace(/\s+/g, " ").trim();
                iframe.contentWindow.postMessage(
                    { type: "jupyterlite-toc-navigate", headingText: headingText },
                    window.location.origin
                );
            });
        });
    }

    /* Material renders the TOC synchronously in the static HTML, so
     * DOMContentLoaded is sufficient; but guard with a MutationObserver
     * in case a future version defers it. */
    attachMaterialTocHandlers();
    if (!document.querySelectorAll('[data-md-component="toc"] a').length) {
        var obs = new MutationObserver(function (_, observer) {
            if (document.querySelectorAll('[data-md-component="toc"] a').length) {
                attachMaterialTocHandlers();
                observer.disconnect();
            }
        });
        obs.observe(document.body, { childList: true, subtree: true });
    }
});
