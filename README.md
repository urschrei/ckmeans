# Ckmeans
Ckmeans clustering is an improvement on heuristic-based clustering
approaches like Jenks. The algorithm was developed in
[Haizhou Wang and Mingzhou Song](http://journal.r-project.org/archive/2011-2/RJournal_2011-2_Wang+Song.pdf) (2011)
as a [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) approach
to the problem of clustering numeric data into groups with the least
within-group sum-of-squared-deviations.

Minimizing the difference within groups – what Wang & Song refer to as
`withinss`, or within sum-of-squares – means that groups are optimally
homogenous within and the data is split into representative groups.
This is very useful for visualization, where you may want to represent
a continuous variable in discrete color or style groups. This function
can provide groups that emphasize differences between data.

Being a dynamic approach, this algorithm is based on two matrices that
store incrementally-computed values for squared deviations and backtracking
indexes.

Unlike the [original implementation](https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html),
this implementation does not include any code to automatically determine
the optimal number of clusters: this information needs to be explicitly
provided.

## Implementation
This is a port (including documentation) of David Schnurr's package <https://github.com/schnerd/ckmeans>, incorporating some improvements from Bill Mill's Python + Numpy implementation at <https://github.com/llimllib/ckmeans>.

# References
1. [Wang, H., & Song, M. (2011). Ckmeans.1d.dp: Optimal k-means Clustering in One Dimension by Dynamic Programming. The R Journal, 3(2), 29.](https://doi.org/10.32614/RJ-2011-015)
2. <https://observablehq.com/@visionscarto/natural-breaks>
