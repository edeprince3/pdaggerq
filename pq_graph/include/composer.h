

#ifndef PDAGGERQ_PRINTER_H
#define PDAGGERQ_PRINTER_H

#include "../../pdaggerq/pq_helper.h"
#include "pq_graph.h"

using namespace pdaggerq;

class Composer {

public:
    Composer() = default;

    /**
     * Compose a string representation of a vertex
     * @param vertex vertex to compose
     * @return string representation of the vertex
     */
    virtual string compose(const VertexPtr &vertex) = 0;

    /**
     * Compose a string representation of a linkage
     * @param linkage linkage to compose
     * @return string representation of the linkage
     */
    virtual string compose(const LinkagePtr &linkage) = 0;

    /**
     * Compose a string representation of a Term
     * @param term term to compose
     * @return string representation of the term
     */
    virtual string compose(const Term &term) = 0;
};


class TAMM_Composer : public Composer {
public:
    TAMM_Composer() = default;

    string compose(const VertexPtr &vertex) override;
    string compose(const LinkagePtr &linkage) override;
    string compose(const Term &term) override;

    /**
     * Compose a string representation of the lines of a vertex
     * @param vertex vertex to compose lines for
     * @param sort whether to sort the lines before composing
     * @return string representation of the lines
     */
    string compose_lines(const VertexPtr &vertex, bool sort = false);
};

#endif //PDAGGERQ_PRINTER_H
