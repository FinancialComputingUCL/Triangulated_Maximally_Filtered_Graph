function [ P, cliques, triangles, peo ] = TMFG_fast(W)
% Base algorithm, uses only T2, builds a random apollonian network.

% INPUT:
%   - W: a weighted network
% OUTPUT:
%   - P: the filtered TMFG graph
%   - cliques: the list of 4-cliques
%   - separators: the list of 3-cliques that are clique separators and also
%     constitute a basis for the cycle space.
%   - junction_tree: one possible junction tree for the graph
%   - peo: perfect elimination ordering

n = size(W,1); % Number of vertices.
P = sparse(n,n); % Sparse matrix.
max_clique_gains = zeros(3*n - 6, 1);
best_vertex = zeros(3*n - 6, 1);

cliques = [];
separators = [];
triangles = [];

% Get first simplex and populate the first 4 triangles in the basis.
cliques(1, :) = max_clique(W);
vertex_list = setdiff(1:n, cliques(1,:));
triangles(1,:) = cliques(1, [1 2 3]); 
triangles(2,:) = cliques(1, [1 2 4]); 
triangles(3,:) = cliques(1, [1 3 4]); 
triangles(4,:) = cliques(1, [2 3 4]); 

peo = cliques(1, :); % Start perfect elimination order with 1st clique.

W(1:(n+1):n^2) = 0;

P(peo, peo) = W(peo, peo);

% Init gain matrix.
for t = 1:4
    [max_clique_gains(t) best_vertex(t)] = get_best_gain(vertex_list, triangles(t,:), W);
end

for i = 1:(n-4)
    % Get maximum local gain.
    [~, nt] = max(max_clique_gains);
    nv = best_vertex(nt);
    %disp(nv);
    
    peo(end + 1) = nv;
    % Add clique.
    cliques(end+1, :) = [nv triangles(nt,:)];
    % Add separators.
    newsep = triangles(nt, :);
    P([nv newsep], [nv newsep]) = W([nv newsep], [nv newsep]);
    separators(end+1, :) = newsep;
    % Replace triangles.
    triangles(nt, :) = [newsep(1) newsep(2) nv];
    % Add two new triangles.
    triangles(end+1, :) = [newsep(1) newsep(3) nv];
    triangles(end+1, :) = [newsep(2) newsep(3) nv];
    % Clean cache of used values.
    vertex_list = setdiff(vertex_list, nv);
    % Update max gains where the vertex nv was involved.
    if length(vertex_list) > 0
        for t = find(best_vertex == nv).'
            [max_clique_gains(t) best_vertex(t)] = get_best_gain(vertex_list, triangles(t,:), W);
        end
    end
    max_clique_gains(nt) = 0;
    ct = size(triangles, 1);
    if length(vertex_list) > 0
        for t = [nt (ct-1) ct]
            [max_clique_gains(t) best_vertex(t)] = get_best_gain(vertex_list, triangles(t,:), W);
        end
    end
end

disp(separators) % Display separators.
disp(cliques) % Display cliques.

end

function [gain vertex] = get_best_gain(vertex_list, triangle, W)
    gvec(vertex_list) = W(vertex_list, triangle(1)) + W(vertex_list, triangle(2)) + W(vertex_list, triangle(3));
    [gain vertex] = max(gvec);
end

function cl = max_clique(W)
    v = sum(W.*(W>mean(W(:))),2);
    [~, sortindex] = sort(v, 'descend');
    cl = sortindex(1:4);
end