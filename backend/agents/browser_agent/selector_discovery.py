"""
Browser Agent - Enhanced Selector Discovery System

Comprehensive DOM analysis that discovers ALL selectors at every nesting level.
Provides the LLM with complete selector maps for robust JavaScript extraction.
"""

import logging
from typing import Dict, Any, List, Optional
from playwright.async_api import Page

logger = logging.getLogger(__name__)


# Enhanced JavaScript discovery script - deep DOM analysis
DEEP_DISCOVERY_SCRIPT = '''() => {
    const discovery = {
        containers: [],           // Repeating container patterns
        selectorMap: {},          // Complete map of selector -> element info
        contentSelectors: {       // Categorized content selectors
            titles: [],
            prices: [],
            ratings: [],
            images: [],
            links: [],
            buttons: []
        },
        dataAttributes: [],       // All data-* attributes found
        recommendedTemplate: null,
        sampleData: []            // Actual extracted sample to prove selectors work
    };
    
    try {
        // ============================================================
        // PHASE 1: Deep Class Frequency Analysis
        // Scan ALL elements and count class occurrences
        // ============================================================
        const classFrequency = new Map();  // class -> {count, depths[], samples[]}
        const tagClassMap = new Map();     // "tag.class" -> count
        
        function analyzeElement(el, depth = 0) {
            if (depth > 15) return;  // Prevent infinite recursion
            
            const tag = el.tagName?.toLowerCase() || '';
            // Handle SVG elements where className is SVGAnimatedString, not a string
            const classStr = typeof el.className === 'string' ? el.className : (el.className?.baseVal || '');
            const classes = classStr.split(' ').filter(c => c.length > 1);
            
            for (const cls of classes) {
                if (!classFrequency.has(cls)) {
                    classFrequency.set(cls, { count: 0, depths: [], samples: [], tags: new Set() });
                }
                const entry = classFrequency.get(cls);
                entry.count++;
                entry.depths.push(depth);
                entry.tags.add(tag);
                if (entry.samples.length < 3) {
                    const text = (el.innerText || '').trim().substring(0, 100);
                    if (text) entry.samples.push(text);
                }
                
                // Track tag.class combinations
                const tagClass = `${tag}.${cls}`;
                tagClassMap.set(tagClass, (tagClassMap.get(tagClass) || 0) + 1);
            }
            
            // Recurse into children
            for (const child of el.children || []) {
                analyzeElement(child, depth + 1);
            }
        }
        
        if (document.body) analyzeElement(document.body);
        
        // ============================================================
        // PHASE 2: Identify Container Patterns (repeating structures)
        // ============================================================
        const containerCandidates = [];
        
        // Find classes that appear 3+ times at similar depths
        for (const [cls, data] of classFrequency) {
            if (data.count >= 3 && data.count <= 100) {
                // Check for consistent depth (items at same level)
                const avgDepth = data.depths.reduce((a, b) => a + b, 0) / data.depths.length;
                const depthVariance = data.depths.reduce((sum, d) => sum + Math.abs(d - avgDepth), 0) / data.depths.length;
                
                if (depthVariance < 2) {  // Items are at similar nesting levels
                    containerCandidates.push({
                        selector: '.' + cls,
                        count: data.count,
                        avgDepth: Math.round(avgDepth),
                        tags: [...data.tags],
                        samples: data.samples.slice(0, 2)
                    });
                }
            }
        }
        
        // Sort by count (most common patterns first)
        containerCandidates.sort((a, b) => b.count - a.count);
        discovery.containers = containerCandidates.slice(0, 10);
        
        // ============================================================
        // PHASE 3: Content Pattern Detection
        // Find selectors for titles, prices, ratings, etc.
        // ============================================================
        
        // PRICE DETECTION - look for currency patterns
        const priceRegex = /^[$€£₹¥₩][\d,.]+|[\d,.]+\s*(USD|EUR|SEK|GBP|INR|kr|:-)|^\$\d/;
        const priceElements = [...document.querySelectorAll('*')].filter(el => {
            if (el.children.length > 2) return false;
            const text = (el.innerText || '').trim();
            return text.length < 30 && priceRegex.test(text);
        });
        
        const priceClassCounts = {};
        priceElements.slice(0, 50).forEach(el => {
            const classStr = typeof el.className === 'string' ? el.className : (el.className?.baseVal || '');
            const classes = classStr.split(' ').filter(c => c.length > 2);
            classes.forEach(cls => {
                priceClassCounts[cls] = (priceClassCounts[cls] || 0) + 1;
            });
            // Also check parent
            const parentClasses = (el.parentElement?.className || '').split(' ').filter(c => c.length > 2);
            parentClasses.forEach(cls => {
                if (cls.toLowerCase().includes('price')) {
                    priceClassCounts[cls] = (priceClassCounts[cls] || 0) + 1;
                }
            });
        });
        
        discovery.contentSelectors.prices = Object.entries(priceClassCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(([cls, count]) => ({
                selector: '.' + cls,
                count,
                sample: priceElements.find(el => el.className.includes(cls))?.innerText?.trim()?.substring(0, 30)
            }));
        
        // TITLE DETECTION - headings and title-like elements
        const titleElements = [...document.querySelectorAll('h1, h2, h3, h4, a[href*="/itm/"], a[href*="/product/"], [class*="title"], [class*="name"], [data-testid*="title"]')]
            .filter(el => {
                const text = (el.innerText || '').trim();
                return text.length > 5 && text.length < 200;
            });
        
        const titleClassCounts = {};
        titleElements.slice(0, 50).forEach(el => {
            const classStr = typeof el.className === 'string' ? el.className : (el.className?.baseVal || '');
            const classes = classStr.split(' ').filter(c => c.length > 2);
            classes.forEach(cls => {
                titleClassCounts[cls] = (titleClassCounts[cls] || 0) + 1;
            });
        });
        
        discovery.contentSelectors.titles = Object.entries(titleClassCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(([cls, count]) => ({
                selector: '.' + cls,
                count,
                sample: titleElements.find(el => el.className.includes(cls))?.innerText?.trim()?.substring(0, 50)
            }));
        
        // RATING DETECTION
        const ratingElements = [...document.querySelectorAll('[class*="rating"], [class*="star"], [class*="review"], [class*="feedback"], [aria-label*="rating"], [aria-label*="star"]')];
        const ratingClassCounts = {};
        ratingElements.slice(0, 30).forEach(el => {
            const classStr = typeof el.className === 'string' ? el.className : (el.className?.baseVal || '');
            const classes = classStr.split(' ').filter(c => c.length > 2);
            classes.forEach(cls => {
                ratingClassCounts[cls] = (ratingClassCounts[cls] || 0) + 1;
            });
        });
        
        discovery.contentSelectors.ratings = Object.entries(ratingClassCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(([cls, count]) => ({
                selector: '.' + cls,
                count,
                sample: ratingElements.find(el => el.className.includes(cls))?.innerText?.trim()?.substring(0, 50)
            }));
        
        // IMAGE DETECTION
        const imageContainers = [...document.querySelectorAll('img, [class*="image"], [class*="img"], [class*="photo"], [class*="thumb"]')]
            .filter(el => {
                const rect = el.getBoundingClientRect();
                return rect.width > 50 && rect.height > 50;
            });
        
        const imageClassCounts = {};
        imageContainers.slice(0, 30).forEach(el => {
            const classStr = typeof el.className === 'string' ? el.className : (el.className?.baseVal || '');
            const classes = classStr.split(' ').filter(c => c.length > 2);
            classes.forEach(cls => {
                imageClassCounts[cls] = (imageClassCounts[cls] || 0) + 1;
            });
        });
        
        discovery.contentSelectors.images = Object.entries(imageClassCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(([cls, count]) => ({ selector: '.' + cls, count }));
        
        // ============================================================
        // PHASE 4: Data Attribute Discovery
        // ============================================================
        const dataAttrs = new Map();
        document.querySelectorAll('*').forEach(el => {
            for (const attr of el.attributes || []) {
                if (attr.name.startsWith('data-') && attr.name.length < 40) {
                    const count = dataAttrs.get(attr.name) || 0;
                    dataAttrs.set(attr.name, count + 1);
                }
            }
        });
        
        discovery.dataAttributes = [...dataAttrs.entries()]
            .filter(([name, count]) => count >= 3)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 15)
            .map(([name, count]) => ({ attribute: name, count }));
        
        // ============================================================
        // PHASE 5: Build Complete Selector Map for Top Container
        // Analyze what's INSIDE the most common container
        // ============================================================
        if (discovery.containers.length > 0) {
            const mainContainer = discovery.containers[0].selector;
            const containerElements = document.querySelectorAll(mainContainer);
            
            if (containerElements.length > 0) {
                const childSelectorCounts = new Map();
                
                // Analyze first 10 containers deeply
                [...containerElements].slice(0, 10).forEach(container => {
                    // Walk all descendants
                    const descendants = container.querySelectorAll('*');
                    descendants.forEach(el => {
                        const tag = el.tagName.toLowerCase();
                        const classStr = typeof el.className === 'string' ? el.className : (el.className?.baseVal || '');
            const classes = classStr.split(' ').filter(c => c.length > 2);
                        const text = (el.innerText || '').trim();
                        
                        // Build selector variations
                        classes.forEach(cls => {
                            const selector = '.' + cls;
                            if (!childSelectorCounts.has(selector)) {
                                childSelectorCounts.set(selector, { 
                                    count: 0, 
                                    tag, 
                                    samples: [],
                                    hasPrice: false,
                                    hasText: false
                                });
                            }
                            const entry = childSelectorCounts.get(selector);
                            entry.count++;
                            if (text && entry.samples.length < 2) {
                                entry.samples.push(text.substring(0, 60));
                            }
                            if (priceRegex.test(text)) entry.hasPrice = true;
                            if (text.length > 10) entry.hasText = true;
                        });
                        
                        // Tag-only selectors for common patterns
                        if (['h1', 'h2', 'h3', 'h4', 'a', 'img', 'span', 'p'].includes(tag)) {
                            const tagSelector = tag;
                            if (!childSelectorCounts.has(tagSelector)) {
                                childSelectorCounts.set(tagSelector, { count: 0, tag, samples: [], hasPrice: false, hasText: false });
                            }
                            childSelectorCounts.get(tagSelector).count++;
                        }
                    });
                });
                
                // Build selector map
                discovery.selectorMap = {
                    container: mainContainer,
                    containerCount: containerElements.length,
                    childSelectors: [...childSelectorCounts.entries()]
                        .filter(([sel, data]) => data.count >= 3)
                        .sort((a, b) => b[1].count - a[1].count)
                        .slice(0, 20)
                        .map(([selector, data]) => ({
                            selector,
                            count: data.count,
                            tag: data.tag,
                            samples: data.samples,
                            likelyPrice: data.hasPrice,
                            hasContent: data.hasText
                        }))
                };
                
                // ============================================================
                // PHASE 6: Generate Working Extraction Template with PROOF
                // ============================================================
                const firstContainer = containerElements[0];
                
                // Try to find actual working selectors by testing them
                const findWorkingSelector = (candidates, testFn) => {
                    for (const candidate of candidates) {
                        const el = firstContainer.querySelector(candidate.selector);
                        if (el && testFn(el)) {
                            return candidate.selector;
                        }
                    }
                    return null;
                };
                
                // Find title selector
                let titleSelector = findWorkingSelector(discovery.contentSelectors.titles, el => el.innerText?.trim()?.length > 5);
                if (!titleSelector) {
                    // Fallback: try common patterns
                    const fallbacks = ['h3', 'h2', 'a', '[class*="title"]', '[class*="name"]'];
                    for (const fb of fallbacks) {
                        const el = firstContainer.querySelector(fb);
                        if (el && el.innerText?.trim()?.length > 5) {
                            titleSelector = fb;
                            break;
                        }
                    }
                }
                
                // Find price selector
                let priceSelector = findWorkingSelector(discovery.contentSelectors.prices, el => priceRegex.test(el.innerText?.trim() || ''));
                if (!priceSelector) {
                    const fallbacks = ['[class*="price"]', 'span'];
                    for (const fb of fallbacks) {
                        const el = firstContainer.querySelector(fb);
                        if (el && priceRegex.test(el.innerText?.trim() || '')) {
                            priceSelector = fb;
                            break;
                        }
                    }
                }
                
                // Find rating selector
                let ratingSelector = findWorkingSelector(discovery.contentSelectors.ratings, el => el.innerText?.trim()?.length > 0);
                if (!ratingSelector) {
                    ratingSelector = '[class*="rating"], [class*="feedback"], [class*="seller"]';
                }
                
                // Build template
                discovery.recommendedTemplate = {
                    container: mainContainer,
                    title: titleSelector || 'h3, h2, a',
                    price: priceSelector || '[class*="price"]',
                    rating: ratingSelector || '[class*="rating"]',
                    code: `return [...document.querySelectorAll('${mainContainer}')].slice(0, 5).map(item => ({
  title: item.querySelector('${titleSelector || "h3, h2, a"}')?.innerText?.trim(),
  price: item.querySelector('${priceSelector || "[class*=price]"}')?.innerText?.trim(),
  rating: item.querySelector('${ratingSelector || "[class*=rating]"}')?.innerText?.trim()
})).filter(x => x.title)`
                };
                
                // PROOF: Actually extract sample data to prove it works
                discovery.sampleData = [...containerElements].slice(0, 3).map(item => ({
                    title: item.querySelector(titleSelector || 'h3, h2, a')?.innerText?.trim()?.substring(0, 80),
                    price: item.querySelector(priceSelector || '[class*="price"]')?.innerText?.trim(),
                    rating: item.querySelector(ratingSelector || '[class*="rating"]')?.innerText?.trim()
                })).filter(x => x.title);
            }
        }
        
    } catch (e) {
        discovery.error = e.message;
    }
    
    return discovery;
}'''


class SelectorDiscovery:
    """Enhanced selector discovery with deep DOM analysis"""
    
    def __init__(self):
        self.cache = {}
        
    async def discover_patterns(self, page: Page) -> Dict[str, Any]:
        """Run comprehensive DOM analysis and return all discovered patterns."""
        try:
            url = page.url
            
            if url in ('about:blank', '') or url.startswith('chrome:'):
                return {}
            
            # Run deep discovery script
            result = await page.evaluate(DEEP_DISCOVERY_SCRIPT)
            
            if result.get('error'):
                logger.warning(f"Discovery error: {result['error']}")
                return {}
            
            # Log discovery results
            container_count = len(result.get('containers', []))
            if container_count > 0:
                logger.info(f"🔍 Discovered {container_count} selector patterns")
                for p in result.get('containers', [])[:3]:
                    logger.info(f"   • {p['selector']} ({p['count']} items)")
                
                # Log child selectors found
                if result.get('selectorMap', {}).get('childSelectors'):
                    child_count = len(result['selectorMap']['childSelectors'])
                    logger.info(f"   📦 Found {child_count} child selectors inside containers")
                
                # Log sample data if extracted
                if result.get('sampleData'):
                    logger.info(f"   ✅ PROOF: Extracted {len(result['sampleData'])} sample items")
            
            return result
            
        except Exception as e:
            logger.warning(f"Selector discovery failed: {e}")
            return {}
    
    def format_for_prompt(self, patterns: Dict[str, Any]) -> str:
        """Format discovered patterns as a comprehensive section for LLM prompt."""
        if not patterns or not patterns.get('containers'):
            return ""
        
        lines = [
            "",
            "═══════════════════════════════════════════════════════════════════════════════",
            "🎯 DISCOVERED SELECTORS (Use these exact selectors for run_js!)",
            "═══════════════════════════════════════════════════════════════════════════════",
            ""
        ]
        
        # Container patterns
        if patterns.get('containers'):
            lines.append("### Item Containers (repeating elements)")
            for p in patterns['containers'][:5]:
                samples = p.get('samples', [])
                sample_preview = f" - e.g. \"{samples[0][:40]}...\"" if samples else ""
                lines.append(f"- `{p['selector']}` ({p['count']} items){sample_preview}")
        
        # Selector map - CRITICAL for extraction
        selector_map = patterns.get('selectorMap', {})
        if selector_map.get('childSelectors'):
            lines.append("")
            lines.append(f"### Inside `{selector_map.get('container')}` containers:")
            for child in selector_map['childSelectors'][:10]:
                indicator = ""
                if child.get('likelyPrice'):
                    indicator = " 💰 PRICE"
                elif child.get('hasContent'):
                    indicator = " 📝 TEXT"
                samples = child.get('samples', [])
                sample = f" → \"{samples[0][:40]}\"" if samples else ""
                lines.append(f"  - `{child['selector']}` ({child['count']}x){indicator}{sample}")
        
        # Content selectors
        content = patterns.get('contentSelectors', {})
        
        if content.get('prices'):
            lines.append("")
            lines.append("### Price Selectors (verified currency patterns)")
            for p in content['prices'][:3]:
                sample = f" → \"{p.get('sample', '')}\"" if p.get('sample') else ""
                lines.append(f"  - `{p['selector']}` ({p['count']}x){sample}")
        
        if content.get('titles'):
            lines.append("")
            lines.append("### Title Selectors")
            for t in content['titles'][:3]:
                sample = f" → \"{t.get('sample', '')[:40]}\"" if t.get('sample') else ""
                lines.append(f"  - `{t['selector']}` ({t['count']}x){sample}")
        
        if content.get('ratings'):
            lines.append("")
            lines.append("### Rating Selectors")
            for r in content['ratings'][:3]:
                sample = f" → \"{r.get('sample', '')[:30]}\"" if r.get('sample') else ""
                lines.append(f"  - `{r['selector']}` ({r['count']}x){sample}")
        
        # Data attributes
        if patterns.get('dataAttributes'):
            lines.append("")
            lines.append("### Data Attributes (stable selectors)")
            for da in patterns['dataAttributes'][:5]:
                lines.append(f"  - `[{da['attribute']}]` ({da['count']}x)")
        
        # Recommended template with PROOF
        tmpl = patterns.get('recommendedTemplate')
        if tmpl:
            lines.append("")
            lines.append("### ⚡ VERIFIED Extraction Template")
            lines.append("```javascript")
            lines.append(tmpl['code'])
            lines.append("```")
        
        # PROOF - Show sample data
        if patterns.get('sampleData'):
            lines.append("")
            lines.append("### 🔬 PROOF - Sample Extracted Data")
            for i, item in enumerate(patterns['sampleData'][:3]):
                lines.append(f"  Item {i+1}: title=\"{item.get('title', 'N/A')[:50]}\", price=\"{item.get('price', 'N/A')}\", rating=\"{item.get('rating', 'N/A')[:20]}\"")
        
        lines.append("")
        lines.append("**USE THESE EXACT SELECTORS** - they are verified to work on this page!")
        lines.append("")
        
        return "\n".join(lines)


# Singleton instance
_discovery = None

def get_selector_discovery() -> SelectorDiscovery:
    """Get the singleton SelectorDiscovery instance."""
    global _discovery
    if _discovery is None:
        _discovery = SelectorDiscovery()
    return _discovery
