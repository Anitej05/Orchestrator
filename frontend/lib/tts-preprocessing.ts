/**
 * Text preprocessing utilities for TTS
 * Converts text to be more natural for speech synthesis
 */

// Number words for conversion
const ONES = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
const SCALES = ['', 'thousand', 'million', 'billion', 'trillion'];

/**
 * Convert a number (0-999) to words
 */
function convertHundreds(num: number): string {
    if (num === 0) return '';

    let result = '';

    if (num >= 100) {
        result += ONES[Math.floor(num / 100)] + ' hundred';
        num %= 100;
        if (num > 0) result += ' ';
    }

    if (num >= 20) {
        result += TENS[Math.floor(num / 10)];
        num %= 10;
        if (num > 0) result += '-' + ONES[num];
    } else if (num > 0) {
        result += ONES[num];
    }

    return result;
}

/**
 * Convert an integer to words
 */
function integerToWords(num: number): string {
    if (num === 0) return 'zero';

    const isNegative = num < 0;
    num = Math.abs(num);

    if (num > 999999999999999) {
        // Too large, return as is
        return (isNegative ? 'negative ' : '') + num.toString();
    }

    const parts: string[] = [];
    let scaleIndex = 0;

    while (num > 0) {
        const chunk = num % 1000;
        if (chunk > 0) {
            const chunkWords = convertHundreds(chunk);
            if (SCALES[scaleIndex]) {
                parts.unshift(chunkWords + ' ' + SCALES[scaleIndex]);
            } else {
                parts.unshift(chunkWords);
            }
        }
        num = Math.floor(num / 1000);
        scaleIndex++;
    }

    return (isNegative ? 'negative ' : '') + parts.join(' ');
}

/**
 * Convert a decimal number to words
 */
function decimalToWords(numStr: string): string {
    const parts = numStr.split('.');
    const intPart = parseInt(parts[0], 10);

    let result = integerToWords(intPart);

    if (parts.length > 1 && parts[1]) {
        result += ' point';
        // Read each digit after decimal point
        for (const digit of parts[1]) {
            result += ' ' + ONES[parseInt(digit, 10)] || 'zero';
        }
    }

    return result;
}

/**
 * Convert numbers in text to words
 */
function convertNumbersToWords(text: string): string {
    // Match numbers including decimals, percentages, currency, etc.
    return text.replace(/(-?\$?\d+(?:,\d{3})*(?:\.\d+)?%?)/g, (match) => {
        // Remove currency symbols and commas for parsing
        let cleaned = match.replace(/[$,]/g, '');
        const hasDollar = match.includes('$');
        const hasPercent = match.includes('%');
        cleaned = cleaned.replace('%', '');

        let result = '';

        if (cleaned.includes('.')) {
            result = decimalToWords(cleaned);
        } else {
            result = integerToWords(parseInt(cleaned, 10));
        }

        if (hasDollar) {
            result += ' dollars';
        }
        if (hasPercent) {
            result += ' percent';
        }

        return result;
    });
}

/**
 * Strip markdown formatting from text
 */
function stripMarkdown(text: string): string {
    let result = text;

    // Remove code blocks (```...```)
    result = result.replace(/```[\s\S]*?```/g, ' code block ');

    // Remove inline code (`...`)
    result = result.replace(/`([^`]+)`/g, '$1');

    // Remove images ![alt](url)
    result = result.replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1');

    // Remove links [text](url) - keep the text
    result = result.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');

    // Remove reference-style links [text][ref]
    result = result.replace(/\[([^\]]+)\]\[[^\]]*\]/g, '$1');

    // Remove headers (# ## ### etc)
    result = result.replace(/^#{1,6}\s+/gm, '');

    // Remove bold/italic (**, __, *, _)
    result = result.replace(/\*\*\*([^*]+)\*\*\*/g, '$1');
    result = result.replace(/\*\*([^*]+)\*\*/g, '$1');
    result = result.replace(/\*([^*]+)\*/g, '$1');
    result = result.replace(/___([^_]+)___/g, '$1');
    result = result.replace(/__([^_]+)__/g, '$1');
    result = result.replace(/_([^_]+)_/g, '$1');

    // Remove strikethrough (~~...~~)
    result = result.replace(/~~([^~]+)~~/g, '$1');

    // Remove horizontal rules
    result = result.replace(/^[-*_]{3,}\s*$/gm, '');

    // Remove blockquotes (>)
    result = result.replace(/^>\s*/gm, '');

    // Remove unordered list markers (-, *, +)
    result = result.replace(/^[\s]*[-*+]\s+/gm, '');

    // Remove ordered list markers (1., 2., etc)
    result = result.replace(/^[\s]*\d+\.\s+/gm, '');

    // Remove task list markers ([ ], [x])
    result = result.replace(/\[[ x]\]\s*/gi, '');

    // Remove HTML tags
    result = result.replace(/<[^>]+>/g, '');

    // Remove table formatting
    result = result.replace(/\|/g, ', ');
    result = result.replace(/^[-:|\s]+$/gm, '');

    // Remove footnotes [^1]
    result = result.replace(/\[\^\d+\]/g, '');

    // Clean up extra whitespace
    result = result.replace(/\n{3,}/g, '\n\n');
    result = result.replace(/[ \t]+/g, ' ');
    result = result.trim();

    return result;
}

/**
 * Convert common abbreviations to full words
 */
function expandAbbreviations(text: string): string {
    const abbreviations: Record<string, string> = {
        'Mr.': 'Mister',
        'Mrs.': 'Missus',
        'Ms.': 'Miss',
        'Dr.': 'Doctor',
        'Prof.': 'Professor',
        'vs.': 'versus',
        'etc.': 'etcetera',
        'e.g.': 'for example',
        'i.e.': 'that is',
        'approx.': 'approximately',
        'est.': 'established',
        'Inc.': 'Incorporated',
        'Ltd.': 'Limited',
        'Corp.': 'Corporation',
        'Jan.': 'January',
        'Feb.': 'February',
        'Mar.': 'March',
        'Apr.': 'April',
        'Jun.': 'June',
        'Jul.': 'July',
        'Aug.': 'August',
        'Sep.': 'September',
        'Sept.': 'September',
        'Oct.': 'October',
        'Nov.': 'November',
        'Dec.': 'December',
    };

    let result = text;
    for (const [abbr, full] of Object.entries(abbreviations)) {
        result = result.replace(new RegExp(abbr.replace('.', '\\.'), 'g'), full);
    }

    return result;
}

/**
 * Clean and format URLs for speech
 */
function cleanUrls(text: string): string {
    // Replace URLs with "link" or extract domain
    return text.replace(/https?:\/\/[^\s]+/g, (url) => {
        try {
            const domain = new URL(url).hostname.replace('www.', '');
            return `link to ${domain}`;
        } catch {
            return 'link';
        }
    });
}

/**
 * Clean special characters that don't speak well
 */
function cleanSpecialCharacters(text: string): string {
    let result = text;

    // Replace common symbols with words
    result = result.replace(/&/g, ' and ');
    result = result.replace(/@/g, ' at ');
    result = result.replace(/#(\w+)/g, 'hashtag $1');
    result = result.replace(/\+/g, ' plus ');
    result = result.replace(/=/g, ' equals ');
    result = result.replace(/->/g, ' to ');
    result = result.replace(/<-/g, ' from ');
    result = result.replace(/\.\.\./g, ', ');
    result = result.replace(/…/g, ', ');

    // Remove remaining special chars that don't add meaning
    result = result.replace(/[<>{}[\]\\^~]/g, ' ');

    // Clean up multiple punctuation
    result = result.replace(/[!?]{2,}/g, '!');
    result = result.replace(/\.{2,}/g, '.');

    return result;
}

/**
 * Main function to preprocess text for TTS
 * Makes text more natural for speech synthesis
 */
export function preprocessTextForTTS(text: string): string {
    if (!text) return '';

    let result = text;

    // Step 1: Strip markdown formatting
    result = stripMarkdown(result);

    // Step 2: Clean URLs
    result = cleanUrls(result);

    // Step 3: Expand abbreviations
    result = expandAbbreviations(result);

    // Step 4: Convert numbers to words
    result = convertNumbersToWords(result);

    // Step 5: Clean special characters
    result = cleanSpecialCharacters(result);

    // Step 6: Final cleanup (collapse spaces/tabs but PRESERVE newlines)
    result = result.replace(/[ \t]+/g, ' ').trim();
    // Ensure max 2 newlines
    result = result.replace(/\n{3,}/g, '\n\n');

    return result;
}
