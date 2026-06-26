// Decoupled per-layer LR: реализовано через shape-based детект в HymOpt::step().
// embed: dims[1] == d_byte → lr_mult_embed (0.5)
// router: dims[0] == num_experts → lr_mult_router (0.5)
// всё остальное: 1.0
