    u = array[x + 0], v = array[x + 2];
    array[x + 0] = u + v;
    array[x + 2] = u - v;
    u = array[x + 1], v = array[x + 3];
    array[x + 1] = u + v;
    array[x + 3] = u - v;
