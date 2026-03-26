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

    function hideOverlay() {
        overlay.classList.add("hidden");
        try {
            iframe.contentWindow.addEventListener("beforeunload", function (e) {
                e.stopImmediatePropagation();
            }, true);
        } catch (_) {
            /* cross-origin — nothing we can do */
        }
    }

    iframe.addEventListener("load", function () {
        setTimeout(hideOverlay, 1000);
    });

    /* Fallback: hide overlay after 10 s even if load event is missed */
    setTimeout(function () { overlay.classList.add("hidden"); }, 10000);
});
