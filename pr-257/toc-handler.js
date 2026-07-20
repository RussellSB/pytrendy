/**
 * TOC Link Handler for JupyterLite iframes
 * 
 * This script intercepts clicks on Table of Contents links and sends a message
 * to the JupyterLite iframe to scroll to the corresponding heading.
 */
(function() {
    'use strict';
    const IFRAME_ORIGIN = window.location.origin;
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initTocHandler);
    } else {
        initTocHandler();
    }
    
    function initTocHandler() {
        const iframe = document.querySelector('iframe[src*="jupyterlite/notebooks"]');
        if (!iframe) {
            console.log('[mkdocs-jupyterlite] No JupyterLite iframe found on this page');
            return;
        }
        console.log('[mkdocs-jupyterlite] TOC handler initialized');
        const tocLinks = document.querySelectorAll('#toc-collapse a[href^="#"]');
        tocLinks.forEach(function(link) {
            link.addEventListener('click', function(event) {
                event.preventDefault();
                const headingText = this.textContent.trim();
                console.log('[mkdocs-jupyterlite] TOC link clicked with text:', headingText);        
                iframe.contentWindow.postMessage({
                    type: 'jupyterlite-toc-navigate',
                    headingText: headingText
                }, IFRAME_ORIGIN);
            });
        });
        console.log('[mkdocs-jupyterlite] Attached handlers to', tocLinks.length, 'TOC links');
    }
})();
