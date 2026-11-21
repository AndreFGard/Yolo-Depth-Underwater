#let results-and-discussions = [
  = Results and Discussions

  == Results

  #let flsea-models = csv("../tables/models_on_flsea.csv")

  #figure(
    {
      set text(size: 9pt, hyphenate: false)
      table(
        columns: (2fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
        align: (left, center, center, center, center),
        inset: (x: 5pt, y: 4pt),
        table.header(..flsea-models.first().map(strong)),
        ..flsea-models.slice(1).flatten(),
      )
    },
    caption: [Avaliação dos modelos no FLSea.],
  ) <tab:flsea-models>

  == Discussion


]
