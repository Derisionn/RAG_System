import { useEffect, useRef, useState } from 'react'
import hljs from 'highlight.js/lib/core'
import sql from 'highlight.js/lib/languages/sql'
import 'highlight.js/styles/github-dark-dimmed.css'

hljs.registerLanguage('sql', sql)

export default function SqlBlock({ code }) {
    const [copied, setCopied] = useState(false)
    const highlightedCode = hljs.highlight(code || '', { language: 'sql' }).value;

    function copy() {
        navigator.clipboard.writeText(code).then(() => {
            setCopied(true)
            setTimeout(() => setCopied(false), 2000)
        })
    }

    return (
        <div className="sql-block">
            <div className="sql-header">
                <span className="sql-label">
                    <span>⬡</span> Generated SQL
                </span>
                <button className={`copy-btn ${copied ? 'copied' : ''}`} onClick={copy}>
                    {copied ? '✓ Copied' : 'Copy'}
                </button>
            </div>
            <div className="sql-code">
                <code className="language-sql" dangerouslySetInnerHTML={{ __html: highlightedCode }}></code>
            </div>
        </div>
    )
}
