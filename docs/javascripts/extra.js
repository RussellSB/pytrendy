/* Disable JupyterLite "Leave site?" prompt on navigation */
document.addEventListener("DOMContentLoaded", function () {
    var iframe = document.getElementById("jupyterlite-iframe");
    if (!iframe) return;

    iframe.addEventListener("load", function () {
        try {
            iframe.contentWindow.addEventListener("beforeunload", function (e) {
                e.stopImmediatePropagation();
            }, true);
        } catch (_) {
            /* cross-origin — nothing we can do */
        }
    });
});
