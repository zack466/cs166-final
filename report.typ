// Overall page settings, similar to LaTeX
#set page(margin: 1in, numbering: "1")
#set par(leading: 0.6em, justify: true)
#set text(font: "Latin Modern Sans", size: 10pt)
#show raw: set text(font: "New Computer Modern Mono")
#show par: set block(spacing: 1.4em)
#show heading: set block(above: 1.4em, below: 1em)
#show link: set text(rgb(0, 0, 255))
// #show figure: set block(breakable: true)
#show figure: it => [#it #v(1em)]

// double spacing
// #set text(top-edge: 0.7em, bottom-edge: -0.3em)
// #set par(leading: 1em)


// Constants
#let COURSENAME = "CS 166 Computational Cameras"
#let AUTHOR = "Zachary Huang";
#let TITLE = "Probabilistic Image Segmentation for Image Colorization";
#let TODAY = datetime.today().display("[month repr:long] [day], [year]");
#let COLOR = gradient.linear(..color.map.rainbow.map(c => c.lighten(60%)))


// Header an all pages but the first
#set page(
  header: context if here().page() > 1 [
    #AUTHOR #h(1fr) #TITLE #h(1fr) #TODAY
    #move(dy: -8pt, line(length: 100%))
  ] else [],
)

// Title block
#block(width: 100%, stroke: 1pt, outset: 8pt, fill: COLOR)[
  *#COURSENAME* #h(1fr) *#TODAY*
  #align(center, text(15pt, strong(TITLE)))
  #AUTHOR
] #v(1em)

= Introduction

Hello there.

= Methods

// short description of algorithm
// describe struggles and solutions

= Results

#pagebreak(weak: true)
#bibliography("../proposal.bib")
