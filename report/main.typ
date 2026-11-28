#import "vendor/charged-ieee/lib.typ": ieee
#import "sections/objectives.typ" : objectives
#import "sections/justificativa.typ": justificativa
#import "sections/modelos.typ": modelos
#import "sections/experiments.typ": experiments
#import "sections/related-work.typ": related-work
#import "sections/results-and-discussions.typ": results-and-discussions
#import "sections/apendice.typ": apendice
#import "sections/eda.typ": eda

#show bibliography: set std.bibliography(title: text(10pt)[References])

#show: ieee.with(
  title: [Finetuning do YOLO-Depth para Estimação Monocular de Profundidade Submarina],
  authors: yaml("authors.yml"),
  bibliography: bibliography("refs.bib", full: true),
)

#objectives
#justificativa
#eda
#modelos
#experiments
#related-work
#results-and-discussions
#apendice
