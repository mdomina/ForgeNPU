module tile_compute_unit_tb;
  localparam int DATA_WIDTH = 8;
  localparam int ACC_WIDTH = 32;
  localparam int ROWS = 2;
  localparam int COLS = 2;
  localparam int DEPTH = 4;
  localparam int ADDR_WIDTH = $clog2(DEPTH);
  localparam int PE_COUNT = ROWS * COLS;

  logic clk;
  logic rst_n;
  logic write_activations_en_i;
  logic [ADDR_WIDTH-1:0] activation_write_addr_i;
  logic signed [ROWS*DATA_WIDTH-1:0] activations_write_data_i;
  logic write_weights_en_i;
  logic [ADDR_WIDTH-1:0] weight_write_addr_i;
  logic signed [COLS*DATA_WIDTH-1:0] weights_write_data_i;
  logic load_vector_en_i;
  logic [ADDR_WIDTH-1:0] activation_read_addr_i;
  logic [ADDR_WIDTH-1:0] weight_read_addr_i;
  logic compute_en_i;
  logic clear_acc_i;
  logic scratchpad_vector_valid_o;
  logic signed [PE_COUNT*ACC_WIDTH-1:0] psums_o;
  logic [PE_COUNT-1:0] valids_o;

  tile_compute_unit #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .write_activations_en_i(write_activations_en_i),
    .activation_write_addr_i(activation_write_addr_i),
    .activations_write_data_i(activations_write_data_i),
    .write_weights_en_i(write_weights_en_i),
    .weight_write_addr_i(weight_write_addr_i),
    .weights_write_data_i(weights_write_data_i),
    .load_vector_en_i(load_vector_en_i),
    .activation_read_addr_i(activation_read_addr_i),
    .weight_read_addr_i(weight_read_addr_i),
    .compute_en_i(compute_en_i),
    .clear_acc_i(clear_acc_i),
    .scratchpad_vector_valid_o(scratchpad_vector_valid_o),
    .psums_o(psums_o),
    .valids_o(valids_o)
  );

  always #5 clk = ~clk;

  task automatic idle_controls;
    begin
      write_activations_en_i = 1'b0;
      write_weights_en_i = 1'b0;
      load_vector_en_i = 1'b0;
      compute_en_i = 1'b0;
      clear_acc_i = 1'b0;
      activations_write_data_i = '0;
      weights_write_data_i = '0;
    end
  endtask

  task automatic write_activation_vector(
    input integer addr,
    input integer row0,
    input integer row1
  );
    begin
      idle_controls();
      activation_write_addr_i = addr[ADDR_WIDTH-1:0];
      activations_write_data_i[0 +: DATA_WIDTH] = row0;
      activations_write_data_i[DATA_WIDTH +: DATA_WIDTH] = row1;
      write_activations_en_i = 1'b1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic write_weight_vector(
    input integer addr,
    input integer col0,
    input integer col1
  );
    begin
      idle_controls();
      weight_write_addr_i = addr[ADDR_WIDTH-1:0];
      weights_write_data_i[0 +: DATA_WIDTH] = col0;
      weights_write_data_i[DATA_WIDTH +: DATA_WIDTH] = col1;
      write_weights_en_i = 1'b1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic load_tile_vectors(
    input integer act_addr,
    input integer weight_addr
  );
    begin
      idle_controls();
      activation_read_addr_i = act_addr[ADDR_WIDTH-1:0];
      weight_read_addr_i = weight_addr[ADDR_WIDTH-1:0];
      load_vector_en_i = 1'b1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic compute_step;
    begin
      idle_controls();
      compute_en_i = 1'b1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic clear_step;
    begin
      idle_controls();
      clear_acc_i = 1'b1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic expect_outputs(
    input integer p0,
    input integer p1,
    input integer p2,
    input integer p3,
    input logic [PE_COUNT-1:0] expected_valids,
    input logic expected_scratchpad_valid
  );
    begin
      if ($signed(psums_o[0 +: ACC_WIDTH]) !== p0 ||
          $signed(psums_o[ACC_WIDTH +: ACC_WIDTH]) !== p1 ||
          $signed(psums_o[2*ACC_WIDTH +: ACC_WIDTH]) !== p2 ||
          $signed(psums_o[3*ACC_WIDTH +: ACC_WIDTH]) !== p3 ||
          valids_o !== expected_valids ||
          scratchpad_vector_valid_o !== expected_scratchpad_valid) begin
        $fatal(1, "tile_compute_unit_tb failed");
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    activation_write_addr_i = '0;
    weight_write_addr_i = '0;
    activation_read_addr_i = '0;
    weight_read_addr_i = '0;
    idle_controls();
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

    write_activation_vector(0, 1, 2);
    write_weight_vector(0, 5, 6);
    write_activation_vector(1, 3, 4);
    write_weight_vector(1, 7, 8);

    load_tile_vectors(0, 0);
    expect_outputs(0, 0, 0, 0, 4'b0000, 1'b1);

    load_tile_vectors(1, 1);
    expect_outputs(0, 0, 0, 0, 4'b0000, 1'b1);

    compute_step();
    expect_outputs(21, 8, 20, 12, 4'b1111, 1'b0);

    compute_step();
    expect_outputs(42, 16, 40, 24, 4'b1111, 1'b0);

    clear_step();
    expect_outputs(0, 0, 0, 0, 4'b0000, 1'b0);

    $display("tile_compute_unit_tb passed");
    $finish;
  end
endmodule
