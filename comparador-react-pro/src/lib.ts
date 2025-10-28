import * as pdfjs from 'pdfjs-dist'
import mammoth from 'mammoth'
import yaml from 'js-yaml'
import { pipeline } from '@xenova/transformers'
import Tesseract from 'tesseract.js'

// worker
import pdfjsWorker from 'pdfjs-dist/build/pdf.worker.mjs?worker'
pdfjs.GlobalWorkerOptions.workerSrc = pdfjsWorker

export type Campo = {
  id: string
  rotulos: string[]
  obrigatorio?: boolean
  tipo?: 'texto' | 'numerico' | 'escolha' | 'booleano'
  valores_validos?: string[]
  min_palavras?: number
  regex?: string
}

export type Schema = {
  nome?: string
  versao?: string
  idioma?: string
  campos: Campo[]
}

export type PageText = {
  pageIndex: number
  fullText: string
  items: { str: string; transform: number[]; width: number; height?: number }[]
  viewport: { width: number; height: number; scale: number }
}

export async function parseYAMLFile(file: File): Promise<Schema>{
  const text = await file.text()
  const obj = yaml.load(text) as Schema
  if (!obj?.campos) throw new Error('Schema YAML sem "campos"')
  return obj
}

export function normalizeText(s: string): string {
  return s.replace(/[ \t]+/g, ' ').replace(/\n{2,}/g, '\n').trim()
}

export async function extractDOCX(file: File): Promise<string>{
  const arrayBuffer = await file.arrayBuffer()
  const { value } = await mammoth.extractRawText({ arrayBuffer })
  return normalizeText(value || '')
}

export async function extractTXT(file: File): Promise<string>{
  const ab = await file.arrayBuffer()
  const dec = new TextDecoder('utf-8')
  return normalizeText(dec.decode(ab))
}

export async function loadPDF(file: File): Promise<{ pdf:any, arrayBuffer:ArrayBuffer }>{ 
  const ab = await file.arrayBuffer()
  const pdf = await pdfjs.getDocument({ data: ab }).promise
  return { pdf, arrayBuffer: ab }
}

export async function extractPerPage(pdf:any, scale=1.75): Promise<PageText[]> {
  const pages: PageText[] = []
  for (let i=1;i<=pdf.numPages;i++){
    const page = await pdf.getPage(i)
    const viewport = page.getViewport({ scale })
    const content = await page.getTextContent()
    const items = content.items.map((it:any)=>({ 
      str: ('str' in it ? it.str : (it?.ts?.s || '')) as string, 
      transform: it.transform, 
      width: it.width, 
      height: (it.height ?? 0) 
    }))
    const fullText = normalizeText(items.map(x=>x.str).join(' '))
    pages.push({ pageIndex: i-1, fullText, items, viewport: { width: viewport.width, height: viewport.height, scale } })
  }
  return pages
}

export async function ocrPDFPages(pdf:any, scale=2): Promise<PageText[]> {
  const pages: PageText[] = []
  for (let i=1;i<=pdf.numPages;i++){
    const page = await pdf.getPage(i)
    const viewport = page.getViewport({ scale })
    const canvas = new OffscreenCanvas(viewport.width, viewport.height) as any
    const ctx = canvas.getContext('2d')
    await page.render({ canvasContext: ctx, viewport } as any).promise
    const blob = await (canvas as any).convertToBlob()
    const ocr = await Tesseract.recognize(blob, 'por+eng')
    const text = normalizeText(ocr.data.text || '')
    pages.push({ pageIndex: i-1, fullText: text, items: [], viewport: { width: viewport.width, height: viewport.height, scale } })
  }
  return pages
}

export function chunkText(s: string, maxLen=600): string[] {
  const words = s.split(/\s+/)
  const out: string[] = []
  let cur: string[] = []
  let len = 0
  for (const w of words){
    cur.push(w)
    len += w.length + 1
    if (len >= maxLen){
      out.push(cur.join(' ')); cur = []; len = 0
    }
  }
  if (cur.length) out.push(cur.join(' '))
  return out
}

let featureExtractor: any = null
export async function getEmbedder() {
  if (!featureExtractor){
    featureExtractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2')
  }
  return featureExtractor
}

export async function embedAll(texts: string[]): Promise<number[][]> {
  const fe = await getEmbedder()
  const outputs = await fe(texts, { pooling: 'mean', normalize: true })
  return outputs.tolist ? outputs.tolist() : outputs
}

export function cosineSim(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0
  for (let i=0;i<a.length;i++){
    dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]
  }
  return dot / (Math.sqrt(na)*Math.sqrt(nb) + 1e-8)
}

export function validateValue(field: any, val: string): string[] {
  const errs: string[] = []
  if (field.obrigatorio && (!val || !val.trim())) errs.push('campo obrigatório não encontrado/preenchido')
  const tipo = field.tipo || 'texto'
  if (tipo === 'numerico'){
    const m = val.match(/[\d\.\,]+/)
    if (!m) errs.push('valor numérico não identificado')
  }
  if (tipo === 'escolha' && field.valores_validos?.length){
    const ok = field.valores_validos.some((v: string) => val.toLowerCase().includes(v.toLowerCase()))
    if (!ok) errs.push('valor não consta na lista permitida')
  }
  if (field.regex){
    const rx = new RegExp(field.regex, 'i')
    if (!rx.test(val)) errs.push('não atende ao padrão (regex)')
  }
  if (field.min_palavras && val){
    if (val.split(/\s+/).length < field.min_palavras) errs.push(`menos que ${field.min_palavras} palavras`)
  }
  return errs
}

export function pickKeywords(text: string, n=6): string[] {
  const stop = new Set(['de','da','do','das','dos','a','o','e','é','ou','para','por','sem','com','em','no','na','nos','nas','um','uma','ao','às','as','os','que','se'])
  const words = text.toLowerCase().replace(/[^\p{L}0-9 ]+/gu,'').split(/\s+/)
  const freq: Record<string, number> = {}
  for (const w of words){
    if (!w || stop.has(w) || w.length < 3) continue
    freq[w] = (freq[w]||0)+1
  }
  return Object.entries(freq).sort((a,b)=>b[1]-a[1]).slice(0,n).map(([w])=>w)
}
