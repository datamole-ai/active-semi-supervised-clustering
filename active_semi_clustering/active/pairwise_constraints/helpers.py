def get_constraints_from_neighborhoods(neighborhoods):
    ml = []

    for neighborhood in neighborhoods:
        for i in neighborhood:
            for j in neighborhood:
                if i != j:
                    ml.append((i, j))

    cl = []
    for neighborhood in neighborhoods:
        for other_neighborhood in neighborhoods:
            if neighborhood != other_neighborhood:
                for i in neighborhood:
                    for j in other_neighborhood:
                        cl.append((i, j))

    return ml, cl
