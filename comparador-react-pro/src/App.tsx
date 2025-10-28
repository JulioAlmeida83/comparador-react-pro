import React, { useEffect, useRef, useState } from 'react'
import { parseYAMLFile, loadPDF, extractPerPage, extractDOCX, extractTXT, ocrPDFPages, embedAll, cosineSim, validateValue, pickKeywords, type Schema } from './lib'

type CampoHit = {
  campo: string
  pageIndex: number
  similaridade: number
  snippet: string
  erros: string[]
  keywords: string[]
}

export default function App(){
  const [yamlFile, setYamlFile] = useState<File|null>(null)
  const [docFile, setDocFile] = useState<File|null>(null)
  const [schema, setSchema] = useState<Schema|null>(null)

  const [pdf, setPdf] = useState<any>(null)
  const [pages, setPages] = useState<Awaited<ReturnType<typeof extractPerPage>>>([])
  const [pageIndex, setPageIndex] = useState<number>(0)

  const [loading, setLoading] = useState(false)
  const [useOCR, setUseOCR] = useState(false)
  const [hits, setHits] = useState<CampoHit[]>([])
  const canvasRef = useRef<HTMLCanvasElement|null>(null)
  const overlayRef = useRef<HTMLDivElement|null>(null)

  async function handleLoadYAML(){
    if (!yamlFile) return
    setSchema(await parseYAMLFile(yamlFile))
  }

  async function handleLoadDoc(){
    if (!docFile) return
    const ext = (docFile.name.split('.').pop()||'').toLowerCase()
    if (ext === 'pdf'){
      const { pdf } = await loadPDF(docFile)
      setPdf(pdf)
      if (useOCR){
        const pgs = await ocrPDFPages(pdf, 2)
        setPages(pgs as any)
      } else {
        const pgs = await extractPerPage(pdf, 1.75)
        setPages(pgs)
      }
      setPageIndex(0)
    } else if (ext === 'docx'){
      const text = await extractDOCX(docFile)
      // monta uma "página única" só pra comparar/visualizar sem canvas
      setPdf(null)
      setPages([{ pageIndex:0, fullText:text, items:[], viewport:{width:800, height:1200, scale:1}} as any])
      setPageIndex(0)
    } else if (ext === 'txt' || ext === 'md'){
      const text = await extractTXT(docFile)
      setPdf(null)
      setPages([{ pageIndex:0, fullText:text, items:[], viewport:{width:800, height:1200, scale:1}} as any])
      setPageIndex(0)
    } else {
      alert('Formato não suportado. Use PDF/DOCX/TXT.')
    }
  }

  async function runCompare(){
    if (!schema || !pages.length) return
    setLoading(true)
    try{
      const pageTexts = pages.map(p => p.fullText)
      const pageEmb = await embedAll(pageTexts)
      const newHits: CampoHit[] = []

      for (const field of schema.campos){
        const q = (field.rotulos||[]).join(' ')
        const [qEmb] = await embedAll([q])
        let bestIdx = 0, bestScore = -1
        for (let i=0;i<pages.length;i++){
          const score = cosineSim(qEmb as number[], pageEmb[i] as number[])
          if (score > bestScore){ bestScore = score; bestIdx = i }
        }
        const pageText = pages[bestIdx].fullText || ''
        const errs = validateValue(field as any, pageText)
        const snippet = pageText.slice(0, 800)
        const keywords = pickKeywords(snippet, 6)
        newHits.push({ campo: field.id, pageIndex: bestIdx, similaridade: Number(bestScore.toFixed(3)), snippet, erros, keywords })
      }
      setHits(newHits)
      if (newHits.length) setPageIndex(newHits[0].pageIndex)
      // desenho do overlay após render
      setTimeout(()=>drawHighlights(), 50)
    } finally {
      setLoading(false)
    }
  }

  // render page to canvas (only if pdf loaded and we have text layer)
  useEffect(()=>{
    (async ()=>{
      if (!pdf || !pages.length) return
      const page = await pdf.getPage(pageIndex+1)
      const viewport = page.getViewport({ scale: (pages[pageIndex] as any).viewport.scale })
      const canvas = canvasRef.current!
      const ctx = canvas.getContext('2d')!
      canvas.width = viewport.width
      canvas.height = viewport.height
      await page.render({ canvasContext: ctx as any, viewport } as any).promise
      drawHighlights()
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageIndex, pdf, pages])

  function drawHighlights(){
    const overlay = overlayRef.current
    if (!overlay) return
    overlay.innerHTML = ''
    if (!pages.length) return
    const page = pages[pageIndex]
    // colete palavras-chave dos hits dessa página
    const kws = new Set<string>()
    for (const h of hits){ if (h.pageIndex === pageIndex){ h.keywords.forEach(k=>k&&k.length>2 && kws.add(k)) } }
    // se não for PDF, não há items para posicionar
    if (!pdf || !(page as any).items?.length || kws.size === 0) return
    for (const it of (page as any).items){
      const s = (it.str||'').toLowerCase()
      if (!s) continue
      let matched = false
      for (const kw of kws){ if (s.includes(kw)){ matched = true; break } }
      if (!matched) continue
      const [a,b,c,d,e,f] = it.transform
      const x = e, y = f - 10
      const w = it.width
      const h = 14
      const div = document.createElement('div')
      div.className = 'hl'
      div.style.left = x + 'px'
      div.style.top = y + 'px'
      div.style.width = Math.max(4, w) + 'px'
      div.style.height = h + 'px'
      overlay.appendChild(div)
    }
  }

  const score = React.useMemo(()=>{
    if (!hits.length) return 0
    const totalPenal = hits.reduce((acc,h)=> acc + h.erros.length, 0)
    return Math.max(0, 100 - 5*totalPenal)
  }, [hits])

  return (
    <div style={{minHeight:'100vh'}}>
      <header style={{padding:'16px 24px', background:'#fff', boxShadow:'0 1px 8px rgba(0,0,0,0.06)'}}>
        <h1 style={{margin:0, fontSize:22}}>Comparador de Documentos — React Pro (100% navegador)</h1>
        <p className="muted" style={{margin:0}}>YAML + (PDF/DOCX/TXT) • Embeddings locais • OCR opcional • Viewer com destaque</p>
      </header>

      <main style={{maxWidth:1280, margin:'0 auto', padding:24}}>
        <section className="row">
          <div className="card">
            <h3 style={{marginTop:0}}>1) Schema (YAML)</h3>
            <input type="file" accept=".yaml,.yml" onChange={e=>setYamlFile(e.target.files?.[0]||null)} />
            <div style={{marginTop:8}}>
              <button className="btn" onClick={handleLoadYAML} disabled={!yamlFile}>Ler YAML</button>
            </div>
            {schema && <p className="muted" style={{marginTop:8}}>Schema: <b>{schema.nome || 'sem nome'}</b> — {schema.campos.length} campos</p>}
          </div>

          <div className="card">
            <h3 style={{marginTop:0}}>2) Documento (PDF/DOCX/TXT)</h3>
            <input type="file" accept=".pdf,.docx,.txt,.md" onChange={e=>setDocFile(e.target.files?.[0]||null)} />
            <label style={{display:'block', marginTop:10, fontSize:14}}>
              <input type="checkbox" checked={useOCR} onChange={(e)=>setUseOCR(e.target.checked)} /> Usar OCR (para PDF escaneado)
            </label>
            <div style={{marginTop:8}}>
              <button className="btn" onClick={handleLoadDoc} disabled={!docFile}>Carregar Documento</button>
            </div>
            {!!pages.length && <p className="muted" style={{marginTop:8}}>{pages.length} página(s) carregadas</p>}
          </div>
        </section>

        <section style={{marginTop:16}}>
          <button className="btn" onClick={runCompare} disabled={!schema || !pages.length || loading}>
            {loading ? 'Processando...' : '3) Comparar'}
          </button>
          {!!hits.length && <span className="badge" style={{marginLeft:12}}>Score: <b>{score}</b></span>}
        </section>

        <section style={{display:'flex', gap:16, marginTop:16}}>
          <aside className="card" style={{width:380}}>
            <h3 style={{marginTop:0}}>Campos & páginas</h3>
            {hits.length === 0 ? <p className="muted">Nenhum resultado ainda.</p> : (
              <ul style={{paddingLeft:18}}>
                {hits.map(h => (
                  <li key={h.campo} style={{marginBottom:10}}>
                    <a href="#" onClick={(e)=>{e.preventDefault(); setPageIndex(h.pageIndex)}}>
                      <b>{h.campo}</b> → pág. {h.pageIndex+1} (sim: {h.similaridade.toFixed(3)})
                    </a>
                    {h.erros.length ? <ul style={{margin:'6px 0 0 18px'}}>{h.erros.map((e,i)=><li key={i} className="err">{e}</li>)}</ul> : <div className="ok">OK</div>}
                  </li>
                ))}
              </ul>
            )}
          </aside>

          <section className="card" style={{flex:1}}>
            {pdf ? (
              <>
                <div style={{display:'flex', alignItems:'center', gap:8, marginBottom:8}}>
                  <button className="btn" onClick={()=>setPageIndex(i=>Math.max(0, i-1))} disabled={pageIndex===0}>◀</button>
                  <div>Página {pageIndex+1} / {pages.length || 0}</div>
                  <button className="btn" onClick={()=>setPageIndex(i=>Math.min((pages.length-1), i+1))} disabled={pageIndex>=pages.length-1}>▶</button>
                </div>
                <div style={{position:'relative'}}>
                  <canvas ref={canvasRef} style={{display:'block'}}/>
                  <div ref={overlayRef} style={{position:'absolute', left:0, top:0}}/>
                </div>
              </>
            ) : (
              <div className="muted">Viewer de PDF aparece aqui quando o documento for PDF. Para DOCX/TXT, a comparação funciona, mas não há visualização.</div>
            )}
          </section>
        </section>
      </main>
    </div>
  )
}
