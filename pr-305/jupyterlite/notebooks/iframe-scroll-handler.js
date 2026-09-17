/**
 * iframe Scroll Handler for JupyterLite
 * 
 * This script listens for postMessage events from the parent page
 * and scrolls to the corresponding heading in the notebook.
 */
(function() {
    'use strict';
    const EXPECTED_ORIGIN = window.location.origin;
    console.log('[mkdocs-jupyterlite] iframe scroll handler loaded');
    window.addEventListener('message', function(event) {
        if (event.origin !== EXPECTED_ORIGIN) {
            console.log('[mkdocs-jupyterlite] Ignoring message from unexpected origin:', event.origin);
            return;
        }
        if (!event.data) {
            console.log('[mkdocs-jupyterlite] Ignoring message with no data');
            return;
        }
        if (event.data.type !== 'jupyterlite-toc-navigate') {
            console.log('[mkdocs-jupyterlite] Ignoring message of unknown type:', event.data.type);
            return;
        }
        
        const headingText = event.data.headingText;
        console.log('[mkdocs-jupyterlite] Received scroll request for:', headingText);

        // We can't use the ID's of the heading elements because jupyterlite uses a different algorithm to generate them
        // from the mkdocs algorithm.
        // For example, the ID of one section in the jupterlite iframe is `Installing-packages-from-PyPI`.
        // But in the mkdocs TOC, the link href is `#installing-packages-from-pypi`.
        
        // If the ID's were consistent, in the iframe we could scroll to the element by ID.
        // We also can't normalize the IDs, because in the outer page they are all
        // lowercase, and in the inner page they are titlecase, so we can't
        // know which words to capitalize.

        // We also can't query all headings and match by text or ID, because
        // jupyterlite does virtual rendering of the notebook (I assume for performance),
        // so cells that are not currently visible in the viewport are not present in the DOM.

        // Instead, we can leverage the JupyterLab TOC extension, which is present in JupyterLite.
        // The TOC extension has a panel with a list of all headings in the notebook,
        // and clicking on an entry in the TOC scrolls to the corresponding heading.
        // So we can find the TOC entry by its title attribute, and simulate a click on it.

        function findTocEntry(title) {
            // Example TOC entry HTML:
            // <span class="jp-tocItem-content" title="Installing packages from PyPI" data-running="-1">Installing packages from PyPI</span>
            // We search for this span by title attribute, and then get its parent link
            const tocSpans = Array.from(document.querySelectorAll('.jp-tocItem-content'));
            for (const span of tocSpans) {
                if (span.getAttribute('title') === title) {
                    return span
                }
            }
            return null;
        }

        function simulateClick(el) {
            // We can't just call el.click() because JupyterLab uses react synthetic events.
            const opts = { view: window, bubbles: true, cancelable: true, buttons: 1 };
            el.dispatchEvent(new MouseEvent('mousedown', opts));
            el.dispatchEvent(new MouseEvent('mouseup', opts));
            el.dispatchEvent(new MouseEvent('click', opts));
        }
    
        function scrollToTocEntry(title) {
            const tocEntry = findTocEntry(title);
            if (tocEntry) {
                console.log('[mkdocs-jupyterlite] Found TOC entry, simulating click to scroll:', title);
                simulateClick(tocEntry);
                return true;
            } else {
                console.log('[mkdocs-jupyterlite] TOC entry not found for title:', title);
                return false;
            }
        }
        let success = scrollToTocEntry(headingText);
        if (success) {
            console.log('[mkdocs-jupyterlite] Found target element via TOC, no further action needed');
        } else {
            // If not found immediately, wait for notebook to fully load and try again
            console.log('[mkdocs-jupyterlite] Target element not found, waiting for notebook to load...');
            const observer = new MutationObserver(function(mutations, obs) {
                const success = scrollToTocEntry(headingText);
                if (success) {
                    console.log('[mkdocs-jupyterlite] Found target element via TOC after mutation, scrolling...');
                    obs.disconnect();
                    return;
                }
            });
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
            // Stop observing after 10 seconds to avoid memory leaks
            setTimeout(function() {
                observer.disconnect();
                console.log('[mkdocs-jupyterlite] Stopped waiting for target element');
            }, 10000);
        }
        
    });
    console.log('[mkdocs-jupyterlite] Message listener attached');
})();
