document.addEventListener("DOMContentLoaded", () => {
  const cards = document.querySelectorAll(".glass-card");
  cards.forEach((card, index) => {
    card.style.opacity = 0;
    card.style.transform = "translateY(20px)";
    setTimeout(() => {
      card.style.transition = "all 0.6s ease";
      card.style.opacity = 1;
      card.style.transform = "translateY(0)";
    }, index * 120);
  });

  const hero = document.querySelector(".hero-title");
  if (hero) {
    hero.classList.add("pulse");
  }

  // Aggressively hide slider min/max bubbles (circular badges)
  function hideSliderBubbles() {
    // Method 1: Hide min/max value containers with inline styles
    const minMaxSelectors = [
      '[data-testid="stSliderMinValue"]',
      '[data-testid="stSliderMaxValue"]',
      '[data-testid="stTickBarMin"]',
      '[data-testid="stTickBarMax"]',
      '[data-testid="stTickBar"]'
    ];

    minMaxSelectors.forEach(selector => {
      document.querySelectorAll(selector).forEach((el) => {
        el.setAttribute('style', 'display: none !important; visibility: hidden !important; opacity: 0 !important; height: 0 !important; width: 0 !important;');
        el.style.display = 'none';
        el.style.visibility = 'hidden';
        el.style.opacity = '0';
        el.style.height = '0';
        el.style.width = '0';
        el.style.overflow = 'hidden';
      });
    });

    // Method 2: Hide all tag elements (bubbles) - more comprehensive search
    const tagSelectors = [
      '[data-baseweb="tag"]',
      '[data-baseweb="slider"] [data-baseweb="tag"]',
      'span[data-baseweb="tag"]',
      'div[data-baseweb="tag"]'
    ];

    tagSelectors.forEach(selector => {
      document.querySelectorAll(selector).forEach((tag) => {
        tag.setAttribute('style', 'display: none !important; visibility: hidden !important; opacity: 0 !important;');
        tag.style.display = 'none';
        tag.style.visibility = 'hidden';
        tag.style.opacity = '0';
      });
    });

    // Method 3: Find elements by text content and style (30 and 100 bubbles)
    const sliders = document.querySelectorAll('[data-testid="stSlider"]');
    sliders.forEach((slider) => {
      // Search in parent containers
      let searchContainer = slider.parentElement;
      for (let level = 0; level < 3 && searchContainer; level++) {
        const allElements = searchContainer.querySelectorAll('*');
        allElements.forEach((child) => {
          const text = child.textContent?.trim();
          if (text === '30' || text === '100') {
            const computedStyle = window.getComputedStyle(child);
            const bgColor = computedStyle.backgroundColor;
            const borderRadius = computedStyle.borderRadius;
            const display = computedStyle.display;

            // If it's not the slider thumb and has visible styling, hide it
            if (
              child.getAttribute('role') !== 'slider' &&
              display !== 'none' &&
              (borderRadius && borderRadius !== '0px' || bgColor && bgColor !== 'rgba(0, 0, 0, 0)' && bgColor !== 'transparent')
            ) {
              child.setAttribute('style', 'display: none !important; visibility: hidden !important; opacity: 0 !important;');
              child.style.display = 'none';
              child.style.visibility = 'hidden';
              child.style.opacity = '0';
            }
          }
        });
        searchContainer = searchContainer.parentElement;
      }
    });

    // Method 4: Find siblings of slider track
    sliders.forEach((slider) => {
      const sliderTrack = slider.querySelector('[data-baseweb="slider"]');
      if (sliderTrack && sliderTrack.parentElement) {
        const siblings = Array.from(sliderTrack.parentElement.children);
        siblings.forEach((sibling) => {
          if (sibling !== sliderTrack) {
            const text = sibling.textContent?.trim();
            const style = window.getComputedStyle(sibling);
            if ((text === '30' || text === '100') &&
                (style.borderRadius !== '0px' || style.backgroundColor !== 'rgba(0, 0, 0, 0)')) {
              sibling.setAttribute('style', 'display: none !important; visibility: hidden !important;');
              sibling.style.display = 'none';
              sibling.style.visibility = 'hidden';
            }
          }
        });
      }
    });
  }

  // Inject CSS dynamically to ensure it loads after Streamlit's CSS
  if (!document.getElementById('slider-bubble-hide-css')) {
    const style = document.createElement('style');
    style.id = 'slider-bubble-hide-css';
    style.textContent = `
      [data-testid="stSliderMinValue"],
      [data-testid="stSliderMaxValue"],
      [data-testid="stTickBarMin"],
      [data-testid="stTickBarMax"],
      [data-testid="stTickBar"],
      [data-baseweb="tag"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
      }
    `;
    document.head.appendChild(style);
  }

  // Run immediately
  hideSliderBubbles();

  // Use MutationObserver to catch dynamically added elements
  const observer = new MutationObserver(() => {
    hideSliderBubbles();
  });

  // Observe the entire document for changes
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['style', 'class', 'data-baseweb']
  });

  // Run multiple times to catch late renders
  setTimeout(hideSliderBubbles, 100);
  setTimeout(hideSliderBubbles, 300);
  setTimeout(hideSliderBubbles, 500);
  setTimeout(hideSliderBubbles, 1000);
  setTimeout(hideSliderBubbles, 2000);
  setTimeout(hideSliderBubbles, 3000);
});
