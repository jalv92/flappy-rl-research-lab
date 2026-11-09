import type { ReactNode } from 'react';
import { Table } from './ui';
// A deliberately small, safe renderer for the server's reports. HTML is always text.
// Supports the headings, lists, tables, code and emphasis used by the report generator.
function inline(text:string):ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part,i)=>part.startsWith('**')&&part.endsWith('**') ? <strong key={i}>{part.slice(2,-2)}</strong> : part.startsWith('`')&&part.endsWith('`') ? <code key={i}>{part.slice(1,-1)}</code> : part);
}
export function Report({ markdown }: { markdown:string }) {
  const lines=markdown.replaceAll('\r\n','\n').split('\n'), blocks:ReactNode[]=[];
  let i=0;
  while(i<lines.length) {
    const line=lines[i], key=i;
    if(!line.trim()){i++;continue;}
    if(line.startsWith('```')) {
      const code:string[]=[];i++;
      while(i<lines.length&&!lines[i].startsWith('```'))code.push(lines[i++]);
      if(i<lines.length)i++;
      blocks.push(<pre key={key}><code>{code.join('\n')}</code></pre>);continue;
    }
    const heading=line.match(/^(#{1,6})\s+(.+)$/);
    if(heading){blocks.push(heading[1].length===1?<h3 key={key}>{inline(heading[2])}</h3>:<h4 key={key}>{inline(heading[2])}</h4>);i++;continue;}
    if(line.startsWith('|')&&lines[i+1]?.match(/^\|[\s:|\-]+\|\s*$/)) {
      const cells=(row:string)=>row.trim().replace(/^\|/,'').replace(/\|$/,'').split('|').map(cell=>cell.trim());
      const headers=cells(line), rows:string[][]=[];i+=2;
      while(i<lines.length&&lines[i].startsWith('|'))rows.push(cells(lines[i++]));
      blocks.push(<Table key={key} headers={headers}>{rows.map((row,r)=><tr key={r}>{row.map((cell,c)=><td key={c}>{inline(cell)}</td>)}</tr>)}</Table>);continue;
    }
    if(/^[-*]\s/.test(line)) {
      const items:string[]=[];
      while(i<lines.length&&/^[-*]\s/.test(lines[i]))items.push(lines[i++].replace(/^[-*]\s/,''));
      blocks.push(<ul key={key}>{items.map((item,k)=><li key={k}>{inline(item)}</li>)}</ul>);continue;
    }
    const text:string[]=[line];i++;
    while(i<lines.length&&lines[i].trim()&&!/^(#|[-*]\s|\||```)/.test(lines[i]))text.push(lines[i++]);
    blocks.push(<p key={key}>{inline(text.join('\n'))}</p>);
  }
  return <article className="report-markdown">{blocks}</article>;
}
