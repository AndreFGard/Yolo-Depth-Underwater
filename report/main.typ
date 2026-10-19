#import "@preview/charged-ieee:0.1.4": ieee
#import "sections/introduction.typ": introduction
#import "sections/methodology.typ": methodology
#import "sections/related-work.typ": related-work
#import "sections/results-and-discussions.typ": results-and-discussions

#show bibliography: set std.bibliography(title: text(10pt)[References])

#show: ieee.with(
  title: [Yolo26 Underwater: YOLOv26-based Object Detection for Underwater Scenarios],
  abstract: [
    Abstrac.s
  ],
  authors: yaml("authors.yml"),
  bibliography: bibliography("refs.bib"),
)

#introduction
#methodology
#related-work
#results-and-discussions
